"""Candidate host scheduling gate for permanently bound output Images.

An unselected graph branch is not assumed unread. The covered native readers,
remaining export-startup gap, and trial release gate are documented in
the product ARCHITECTURE.md native-reader limitations. No render-engine identity participates in admission.
"""
from contextlib import contextmanager
from collections import deque
import threading

import bpy as _native_bpy


JOBS = ('RENDER', 'RENDER_PREVIEW', 'OBJECT_BAKE', 'COMPOSITE', 'SHADER_COMPILATION')
# These native jobs have no public is_job_running enum. Observe their owning
# modal operator; light bake installs it before worker start and cancels by
# joining the worker. Multires bake is conservatively deferred too.
MODAL_READERS = frozenset({'OBJECT_OT_lightprobe_cache_bake', 'OBJECT_OT_bake_image'})
from types import SimpleNamespace
from ..integration.components import Component
from ...blender_runtime import PendingTransitionError
from ...blender_runtime.policies import native_evaluation as policy

def create(_application):
    bpy = _native_bpy
    _state = SimpleNamespace(_publishing=False, _callbacks=0, _operations=0)
    # Constructed on the foreground path before native handlers are registered.
    # The snapshot is immutable actor output, never a second mutable automaton.
    store = _application.app.store(machine=_application.machines['native_evaluation'], initial={'tokens': ()})
    snapshot = store.state
    lock = threading.Lock()
    ingress = deque()
    routes = {}
    generation = snapshot.generation
    revision = snapshot.revision
    serial = snapshot.serial
    pending_transition = None
    fault = None
    reconciling = False
    last_available = True


    class PublicationDeferred(RuntimeError):
        """Native readers currently prevent a payload installation."""


    def reason(*, cleanup=False):
        """Check native readers; teardown may revisit a fenced App's handles."""
        if not cleanup and _application.waiting_for_readers:
            return 'Output installation is waiting for application cleanup'
        if not cleanup and (_application.app.fenced or _application.app.closed):
            return 'Output installation belongs to a retired application'
        if threading.current_thread() is not threading.main_thread():
            return 'Output installation requires Blender\'s main thread'
        if _state._publishing:
            return 'An output installation is already active'
        if _state._callbacks:
            return 'Output installation is deferred from a native evaluation callback'
        if _state._operations:
            return 'A synchronous native output operation is active'
        with lock:
            observed_generation, observed_revision = generation, revision
        if fault is not None:
            return 'Native evaluation tracking failed; runtime recovery is required'
        if not policy.available(snapshot, observed_generation, observed_revision):
            return 'Native evaluation has not finished'
        blocked = native_reason()
        if blocked is not None:
            return blocked
        # A callback can arrive while Blender's live job/lock observations are
        # collected. Never return the older acknowledged readiness afterward.
        with lock:
            if (_state._publishing or _state._callbacks or _state._operations
                    or not policy.available(snapshot, generation, revision)):
                return 'Native evaluation has not finished'
        return None


    def native_reason():
        """Only physical observations; never actor work or graph evaluation."""
        if not hasattr(bpy.data, 'window_managers'):
            return 'Output installation is waiting for Blender data'
        if any(manager.is_interface_locked for manager in bpy.data.window_managers):
            return 'Output installation is waiting for Blender\'s interface lock'
        for manager in bpy.data.window_managers:
            for window in manager.windows:
                for operator in window.modal_operators:
                    if operator.bl_idname in MODAL_READERS:
                        return 'Output installation is waiting for Blender operation ' + operator.bl_idname
        for job in JOBS:
            if bpy.app.is_job_running(job):
                return 'Output installation is waiting for Blender job ' + job
        return None


    def available(*, cleanup=False):
        return reason(cleanup=cleanup) is None


    def require_available():
        # Explicit effects also run in background scripts without a UI timer.
        # Reconcile refuses native callbacks and active publication/operations;
        # passive reason()/available() never enter the actor bridge.
        reconcile()
        blocked = reason()
        if blocked is not None:
            raise PublicationDeferred(blocked)


    def publishing():
        return _state._publishing


    def evaluation_started(kind, token=0):
        """Latch a native start and revoke observations without waiting on actors."""
        nonlocal serial, revision
        with lock:
            serial += 1
            revision += 1
            receipt = (generation, serial)
            routes.setdefault((kind, int(token)), deque()).append(receipt)
            ingress.append(('begin', generation, revision, serial))
        return receipt


    def evaluation_finished(kind, token=0):
        """Capture a terminal receipt; its actor commit belongs to the foreground.

        Blender supplies a scene, not an operation ID. Same-scene native events
        are paired in order under its balanced callback contract. Captured
        receipts retain their original generation across native replacement.
        """
        nonlocal revision
        with lock:
            key = (kind, int(token))
            leases = routes.get(key)
            if not leases:
                return
            lease_generation, lease = leases.popleft()
            if not leases:
                routes.pop(key)
            revision += 1
            ingress.append(('finish', lease_generation, revision, lease))


    def reconcile(*, cleanup=False):
        """Commit callback observations only from an ordinary foreground turn.

        A timed-out actor operation retains its original future and event. Never
        resubmit it, or mistake the timeout for native completion.
        """
        nonlocal snapshot, pending_transition, fault, reconciling, last_available
        if (threading.current_thread() is not threading.main_thread() or reconciling
                or _state._callbacks or _state._publishing or _state._operations
                or fault is not None or _application.app.closed
                or (_application.app.fenced and not cleanup)):
            return False
        reconciling = True
        try:
            if pending_transition is not None:
                stage, token = pending_transition
                host = _application.app.actors
                pending = host.pending.get(token)
                if pending is not None and not pending[1].done():
                    return False
                result = host.reconcile(token)
                pending_transition = None
                if stage == 'dispatch':
                    if not result.consumed:
                        raise RuntimeError('Native evaluation observation lost its ordering')
                    with lock:
                        ingress.popleft()
                    try:
                        snapshot = store.state
                    except PendingTransitionError as error:
                        pending_transition = ('snapshot', error.token)
                        return False
                else:
                    snapshot = result
            # Bounded foreground turns preserve responsiveness during playback.
            for _ in range(64):
                with lock:
                    observed = ingress[0] if ingress else None
                if observed is None:
                    break
                kind, event_generation, event_revision, lease = observed
                index = snapshot.tokens.index(lease) if lease in snapshot.tokens else -1
                proposed = (snapshot.tokens + (lease,) if kind == 'begin' else
                            snapshot.tokens[:index] + snapshot.tokens[index + 1:]
                            if kind == 'finish' and index >= 0 else
                            () if kind == 'replace' else snapshot.tokens)
                event = policy.Event(kind, generation=event_generation, revision=event_revision,
                                     lease=lease, index=index, proposed_tokens=proposed,
                                     validated=kind in ('finish', 'replace'))
                try:
                    effect = (store._dispatch_cleanup(event) if cleanup else store.dispatch(event))
                except PendingTransitionError as error:
                    pending_transition = ('dispatch', error.token)
                    return False
                if not effect.consumed:
                    raise RuntimeError('Native evaluation observation lost its ordering')
                with lock:
                    ingress.popleft()
                try:
                    snapshot = store.state
                except PendingTransitionError as error:
                    pending_transition = ('snapshot', error.token)
                    return False
            current = available(cleanup=cleanup)
            if current != last_available:
                last_available = current
                for screen in getattr(bpy.data, 'screens', ()):
                    for area in screen.areas:
                        if area.type == 'NODE_EDITOR':
                            area.tag_redraw()
            return current
        except PublicationDeferred:
            return False
        except Exception as error:
            fault = error
            raise
        finally:
            reconciling = False


    def reset_after_native_replacement():
        """Discard old evaluation tokens only after a completed host replacement."""
        nonlocal generation, revision
        if (threading.current_thread() is not threading.main_thread()
                or _state._publishing or _state._callbacks or _state._operations):
            raise PublicationDeferred('Native replacement recovery is not quiescent')
        blocked = native_reason()
        if blocked is not None:
            raise PublicationDeferred(blocked)
        # The recovery coordinator calls this only after its native post receipt
        # and the physical job checks above. Retire old native routing identities;
        # already captured receipts in ingress retain their original generation.
        with lock:
            revision += 1
            ingress.append(('replace', generation, revision, 0))
            generation += 1
            routes.clear()
        reconcile(cleanup=True)


    @contextmanager
    def publication():
        """Enclose transfer, verification, scalar selection, and notification.

        Code inside this interval must not invoke operators, evaluate a depsgraph,
        change frames, pump events, or yield to another callback.
        """
        require_available()
        _state._publishing = True
        try:
            yield
        finally:
            _state._publishing = False


    @contextmanager
    def native_callback():
        """Record synchronous callbacks, including background render handlers."""
        with lock:
            _state._callbacks += 1
        try:
            yield
        finally:
            with lock:
                _state._callbacks -= 1


    @contextmanager
    def native_operation():
        """Cover synchronous native invocations not reported as running WM jobs."""
        require_available()
        _state._operations += 1
        try:
            yield
        finally:
            _state._operations -= 1
            reconcile(cleanup=_application.app.fenced)

    def observation():
        """Read-only acknowledged machine state for diagnostics and tests."""
        return snapshot

    def _set_export_reconcile(value):
        nonlocal reconcile
        reconcile = value

    def _set_export_PublicationDeferred(value):
        nonlocal PublicationDeferred
        PublicationDeferred = value

    def _set_export_reason(value):
        nonlocal reason
        reason = value

    def _set_export_available(value):
        nonlocal available
        available = value

    def _set_export_require_available(value):
        nonlocal require_available
        require_available = value

    def _set_export_publishing(value):
        nonlocal publishing
        publishing = value

    def _set_export_evaluation_started(value):
        nonlocal evaluation_started
        evaluation_started = value

    def _set_export_evaluation_finished(value):
        nonlocal evaluation_finished
        evaluation_finished = value

    def _set_export_reset_after_native_replacement(value):
        nonlocal reset_after_native_replacement
        reset_after_native_replacement = value

    def _set_export_publication(value):
        nonlocal publication
        publication = value

    def _set_export_native_callback(value):
        nonlocal native_callback
        native_callback = value

    def _set_export_native_operation(value):
        nonlocal native_operation
        native_operation = value

    def _set_export_bpy(value):
        nonlocal bpy
        bpy = value

    return Component({
        'bpy': bpy,
        'PublicationDeferred': PublicationDeferred,
        'reason': reason,
        'available': available,
        'require_available': require_available,
        'publishing': publishing,
        'evaluation_started': evaluation_started,
        'evaluation_finished': evaluation_finished,
        'reconcile': reconcile,
        'observation': observation,
        'reset_after_native_replacement': reset_after_native_replacement,
        'publication': publication,
        'native_callback': native_callback,
        'native_operation': native_operation,
        'JOBS': JOBS,
        'MODAL_READERS': MODAL_READERS,
    }, state=_state, writers={'bpy': _set_export_bpy, 'PublicationDeferred': _set_export_PublicationDeferred, 'reason': _set_export_reason, 'available': _set_export_available, 'require_available': _set_export_require_available, 'publishing': _set_export_publishing, 'evaluation_started': _set_export_evaluation_started, 'evaluation_finished': _set_export_evaluation_finished, 'reconcile': _set_export_reconcile, 'reset_after_native_replacement': _set_export_reset_after_native_replacement, 'publication': _set_export_publication, 'native_callback': _set_export_native_callback, 'native_operation': _set_export_native_operation})
