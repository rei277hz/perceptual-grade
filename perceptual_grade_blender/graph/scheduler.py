"""Blender adapter for the executable scheduling automaton.

Only this module translates RNA identities, gestures and immutable snapshots
into policy tokens. Numerical workers keep their own payloads; they cannot
authorize publication. Timer registration also passes through this module.
"""

from ..integration.components import Component

def create(_application):
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text
    from collections import OrderedDict
    from collections.abc import Mapping
    from dataclasses import dataclass
    from itertools import count
    from uuid import uuid4

    from ...perceptual_grade_core.graph import scheduler_policy as policy
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    from ...blender_runtime.execution.diagnostics import log_exception

    _serial = count(1)
    _session = uuid4().int
    _records = {}
    _gestures = {}
    _keys = OrderedDict()
    _timers = set()
    _timer_boundaries = {}
    _updates = OrderedDict()
    _frames = set()
    _frame_signals = {}
    _commits = {}
    _displayed = {}
    _preview_turn = {}
    _lifecycle_store = _application.app.store(machine=_application.machines['lifecycle'])


    @dataclass(frozen=True)
    class Ticket:
        channel: tuple
        owner: int
        key: int
        signal: int
        quality: int
        token: int
        session: int = 1
        resolution: int = 0
        provenance: tuple = ()
        provenance_token: int = 0
        epoch: int = 0
        # Frozen numerical evaluator/model identity.  This is separate from the
        # source signal: control edits may leave an old coherent result visible
        # while a new evaluator instance is in flight.
        evaluation: tuple = ()
        display_source: int = 0
        display: tuple = ()
        application: str = ''
        history: bool = False


    class ScopePayloadError(ValueError):
        """A worker returned a scope envelope that cannot be displayed.

        This is deliberately separate from a stale ticket: a finished worker may
        legitimately lose admission after a source edit, while a payload carrying
        a preview-resolution cube is an adapter/data-integrity failure that needs
        a visible Retry state.
        """


    @dataclass(frozen=True)
    class EffectLease:
        channel: tuple
        owner: int
        session: int
        token: int
        phase: int
        success: object
        failure: object


    EFFECT_PHASES = ('capture', 'submit', 'worker', 'gpu', 'upload', 'install',
                     'timer')


    def session():
        return _session


    def lifecycle_event(kind):
        """Receipt-driven save/load protocol; native handlers execute its effects."""
        if kind == "recovered":
            ownership = _application.component('ownership')
            ownership.require_ready()
        effects=_lifecycle_store.dispatch(policy.LifecycleEvent(kind))
        if not effects.accept:
            raise RuntimeError(f'Invalid scheduler lifecycle event {kind} in phase {_lifecycle_store.state.phase}')
        return _lifecycle_store.state


    def lifecycle_admits(action):
        if _application.app.fenced or _application.waiting_for_readers:
            return False
        ownership = _application.component('ownership')
        transactions = _application.component('ownership.transactions')
        return bool(_lifecycle_store.query('lifecycle_admits', action=action)
                    and ownership.admits(action)
                    and (action in ('timer', 'rebind') or not transactions.pending()))


    def identity(value):
        key = repr(value)
        if key not in _keys:
            _keys[key] = next(_serial)
        _keys.move_to_end(key)
        result = _keys[key]
        if len(_keys) > 4096:
            pinned = {token for record in _records.values() for token in
                      (record['store'].state.requested,record['store'].state.signal,record['store'].state.worker_key)}
            pinned.update(token for record in _records.values()
                for entry in record.get('scope_slots',{}).values()
                for token in vars(entry['store'].state).values())
            for old, token in tuple(_keys.items()):
                if len(_keys) <= 4096:break
                if token not in pinned:_keys.pop(old)
        return result


    def output(node):
        editing = _application.component('graph.editing')
        return node.stack if editing.is_target(node) else node


    def _affected(node, edited):
        grades = _application.component('graph.grades'); stacks = _application.component('graph.stacks')
        owner = output(node)
        if owner == output(edited):
            return True
        grade = grades.selected(edited)
        if grade is None:
            return False
        if stacks.is_stack(owner):
            return any(item.enabled and item.grade == grade for item in owner.pg_stack_entries)
        return grades.selected(owner) == grade


    def _scope_affected(node, edited):
        """Follow only signals sampled by this scope, without evaluating them."""
        nodes = _application.component('graph.nodes'); stacks = _application.component('graph.stacks')
        reference = _application.component('scopes.reference')
        owner = output(node)
        if _affected(owner, edited):return True
        pending = [owner.inputs.get('Color')]
        if reference.visible(node):pending.append(node.inputs.get(reference.SOCKET))
        seen = {owner.as_pointer()}
        while pending:
            socket = pending.pop()
            if socket is None or not socket.is_linked:continue
            link = socket.links[0]
            source = link.from_node
            pointer = source.as_pointer()
            if pointer in seen:continue
            seen.add(pointer)
            if source.bl_idname == 'NodeReroute':
                pending.append(source.inputs[0])
            elif (nodes.is_grade(source) or stacks.is_stack(source)) and link.from_socket.name == 'Color':
                if _affected(source, edited):return True
                # An upstream Grade's comparison input does not affect its output.
                pending.append(source.inputs.get('Color'))
            else:
                # Blender evaluates arbitrary intermediate nodes. Gesture
                # discovery follows their graph edges without interpreting
                # individual node operations.
                pending.extend(source.inputs)
                tree = getattr(source,'node_tree',None)
                if tree is not None:
                    for endpoint in tree.nodes:
                        if endpoint.bl_idname=='NodeGroupOutput' and endpoint.is_active_output:
                            pending.extend(endpoint.inputs)
        return False


    def _gesture_matches(node, affected):
        if node is None:return False
        for edited_ref in tuple(_gestures.values()):
            edited = resolve(edited_ref)
            if edited is None:continue
            try:
                if affected(node, edited):
                    return True
            except (ReferenceError, RuntimeError):
                continue
        return False


    def dragging(node):
        """An edit to this output's own Grade or enabled Stack entries."""
        return _gesture_matches(node, _affected)


    def scope_dragging(node):
        """Include edits in the primary and visible comparison signal chains."""
        return _gesture_matches(node, _scope_affected)


    def _record_dragging(record):
        check = scope_dragging if record['store'].state.scope else dragging
        return check(resolve(record['node']))


    def _event(record, kind, **fields):
        return record['store'].dispatch(policy.Event(kind,
            owner=record['store'].state.owner, session=record['store'].state.session, **fields))


    def _channel_store(*, scope=False, linear=False):
        return _application.app.store(machine=_application.machines['channel'], initial={
            'owner': next(_serial), 'key': 0, 'session': _session,
            'scope': scope, 'linear': linear})


    def _ticket_current(ticket):
        return type(ticket) is Ticket and ticket.application == _application.app.identity


    def _record(node, kind, context, channel=None):
        channel = kind, node.as_pointer() if channel is None else channel
        record = _records.get(channel)
        if record is None or record['context'] != context:
            previous = None
            previous_display = ()
            if record is not None:
                if kind in ('linear','full') and isinstance(channel[1],tuple):
                    display = _application.component('scopes.display')
                    previous = (display.result(record) if 'scope_slots' in record else record.get('result'))
                    if previous is None:previous = record.get('scope_previous')
                    previous_display = record.get('scope_visible',record.get('scope_previous_display',()))
                _event(record, 'shutdown')
            record = dict(store=_channel_store(scope=kind in ('linear', 'full'),linear=kind=='linear'), context=context, node=retain(node,'scheduler.record'))
            if previous is not None:
                # Retain one complete measurement while a new native source is
                # evaluated. Its old owner is retired and cannot publish again.
                record.update(scope_previous=previous,scope_previous_display=previous_display)
            _records[channel] = record
        active = _record_dragging(record)
        if active != record['store'].state.dragging:
            _event(record, 'begin' if active else 'release')
        return channel, record


    def begin(node):
        token = next(_serial)
        _gestures[token] = retain(node,'scheduler.gesture')
        for record in tuple(_records.values()):
            if _record_dragging(record):
                _event(record, 'begin')
        return token


    def end(token):
        edited = resolve(_gestures.pop(token, None))
        if edited is None:
            return
        for record in tuple(_records.values()):
            if not _record_dragging(record):
                _event(record, 'release')
        nodes = _application.component('graph.nodes'); lut = _application.component('graph.lut')
        # A Grade can occur in several output nodes, including inverse Stacks.
        for node in nodes.all_nodes():
            if _affected(node, edited) and not dragging(node):
                lut.settle(node)


    def lut_invalidated(node):
        """An invalid payload must be publishable again even for the same model."""
        record=_records.pop(('lut',node.as_pointer()),None)
        if record is not None:_event(record,'shutdown')


    def lut_request(node, model):
        lut = _application.component('graph.lut')
        channel, record = _record(node, 'lut', lut.context(node))
        key = identity(lut.cm.signature(model))
        _event(record, 'request', key=key)
        signal = identity(lut.context(node))
        _event(record, 'signal', signal=signal)
        return Ticket(channel, record['store'].state.owner, key, signal, 0, next(_serial), _session, application=_application.app.identity)


    def lut_candidate(node, model, preview):
        lut = _application.component('graph.lut'); nodes = _application.component('graph.nodes')
        lut_request(node, nodes.bake_model(node))
        record = _records['lut', node.as_pointer()]
        key = identity(lut.cm.signature(model))
        # The signal token for LUT publication denotes its output context; only
        # scopes depend on an installed image token. The key checks model currency.
        return Ticket(('lut', node.as_pointer()), record['store'].state.owner, key,
                      identity(lut.context(node)), 1 if preview else 2, next(_serial), _session, application=_application.app.identity)


    def lut_cache_hit(node, model):
        ticket = lut_candidate(node, model, False)
        ready(ticket, True)
        publication(ticket)


    def publication(ticket, commit=True):
        if not _ticket_current(ticket):return False
        if not lifecycle_admits('publish'):return False
        record = _records.get(ticket.channel)
        if record is None or ticket.session != _session:
            return False
        event = policy.Event('publish',
            owner=ticket.owner, session=ticket.session, key=ticket.key,
            signal=ticket.signal, quality=ticket.quality,
            resolution=ticket.resolution, provenance=ticket.provenance_token,epoch=ticket.epoch)
        effects = record['store'].dispatch(event) if commit else record['store'].preview(event).effects
        return bool(effects.accept)


    def ready(ticket, value):
        if not _ticket_current(ticket):return None
        record = _records.get(ticket.channel)
        if (record is not None and record['store'].state.owner == ticket.owner
                and ticket.session == _session):
            _event(record, 'ready', ready=value)


    def replacement(node, key, started, preview, finished=False, published=False,
                    cancelled=False, scope_demand=False):
        record = _records['lut', node.as_pointer()]
        transition = record['store'].preview(policy.Event('replace',
            owner=record['store'].state.owner, key=identity(key), started=started,
            preview=preview, finished=finished, published=published,
            cancelled=cancelled, scope_demand=scope_demand))
        return transition.effects


    def scope_request(node, kind, spec):
        display = _application.component('scopes.display')
        display_source = spec.get('_scope_display_source')
        channel, record = _record(node, kind, display_source if display_source is not None else spec['context'],
                                  channel=spec.get('_scope_channel'))
        provenance = tuple(tuple(item) for item in spec.get('scope_provenance', ()))
        resolution = int(spec.get('scope_resolution', 0))
        provenance_token = identity(provenance)
        evaluation = spec.get('_scope_evaluation')
        if evaluation is not None:
            evaluation = tuple(evaluation)
        # Stage-1 is a separate matrix-only contract and predates the full
        # analytical evaluator envelope. Its payload has no full-scope identity.
        ticket_evaluation = (evaluation or ()) if kind == 'full' else ()
        if kind == 'linear':
            key, signal, quality = policy.linear_scope_request(
                identity(spec['key']), identity(spec.get('scope_source', spec['context'])))
        else:
            # Source/context changes fence a capture. Control changes remain in its
            # request key, so a frozen final scope may finish during a drag.
            # The spec owns the evaluator identity. Reading RNA here would let a
            # mutable post-capture model redefine an in-flight request.
            if evaluation is None:
                raise ValueError('Scope request has no frozen evaluation identity')
            semantic = spec['key'].scheduler_projection(evaluation)
            signal = identity(spec.get('_scope_compatibility',
                                       spec.get('scope_source', (spec['context'], spec['key'].source_projection()))))
            quality = 2
            key = identity(semantic)
        _event(record, 'request', key=key, provenance=provenance_token,history=bool(spec.get('_scope_history')))
        if display_source is None and record['store'].state.signal != signal:
            record.pop('result',None);record.pop('display_ticket',None)
        _event(record, 'signal', signal=signal)
        ticket = Ticket(channel, record['store'].state.owner, key, signal, quality, next(_serial),
                      _session, resolution, provenance, provenance_token,record['store'].state.epoch,
                      ticket_evaluation, identity((record['store'].state.owner, _session, display_source))
                      if display_source is not None else 0,
                      display.descriptors(spec, kind, identity) if display_source is not None else (),
                      application=_application.app.identity,history=bool(spec.get('_scope_history')))
        if ticket.display:
            if tuple(row[0] for row in ticket.display) != record.get('scope_previous_display'):
                record.pop('scope_previous',None)
            display.request(record, ticket)
        return ticket


    def scope_coherent(node, kind, spec):
        token = spec.get('_schedule')
        if not _ticket_current(token):return False
        record = _records.get(token.channel)
        if record is None:return False
        # Final provenance is obtained by capture, never inferred from the
        # currently installed (possibly 33³ preview) image.
        # Stack routes consume a captured final cube. Regular Grades carry their
        # analytical model and may legitimately have no baked provenance.
        analytical = bool(token.evaluation) and token.evaluation[0] != 'stack-final-lut'
        coherent = token.resolution == 65 and (
            (analytical and not token.provenance) or _final_provenance(token.provenance))
        return bool(coherent and token.session == _session and
                    record['store'].state.owner == token.owner and record['store'].state.live and
                    record['store'].state.signal == token.signal)


    def _final_provenance(provenance):
        try:
            return all(len(item) == 2 for item in provenance) and policy.final_scope_resolution(
                tuple(item[1] for item in provenance))
        except (TypeError, ValueError, OverflowError):
            return False


    def scope_start(ticket):
        if not _ticket_current(ticket):return False
        if not lifecycle_admits('start'):return False
        record = _records.get(ticket.channel)
        if record is None:
            return False
        effects = record['store'].dispatch(policy.Event('start',
            owner=ticket.owner, session=ticket.session, key=ticket.key,
            token=ticket.token, resolution=ticket.resolution,
            provenance=ticket.provenance_token,epoch=ticket.epoch,history=ticket.history))
        return bool(effects.start)


    def scope_retry(node, kind='full', channel=None):
        """Explicitly rearm the failed current scope request, retaining its display."""
        if channel is None:channel = _application.component('scopes.sampling').channel(node)
        record = _records.get((kind, channel))
        if record is None:return False
        state = record['store'].state
        effects = record['store'].dispatch(policy.Event('retry',
            owner=state.owner, session=state.session, key=state.requested))
        if not effects.accept:return False
        record.pop('error', None)
        return True


    def scope_complete(ticket):
        if not _ticket_current(ticket):return None
        record = _records.get(ticket.channel)
        if record is not None:
            record['store'].dispatch(policy.Event('complete',
                owner=ticket.owner, session=ticket.session, token=ticket.token))


    def scope_bake_request(signature):
        """Independent final-cube demand; never supersedes the native preview."""
        key=identity(signature);channel=('scope_bake',key)
        record=_records.get(channel)
        if record is None:
            record=dict(store=_channel_store(scope=True),
                        context=signature,node=None)
            _records[channel]=record
        provenance=((signature,65),);provenance_token=identity(provenance)
        _event(record,'request',key=key,provenance=provenance_token)
        _event(record,'signal',signal=key)
        return Ticket(channel,record['store'].state.owner,key,key,2,next(_serial),_session,65,
                      provenance,provenance_token,record['store'].state.epoch, application=_application.app.identity)


    def scope_bake_clear(signature):
        record=_records.pop(('scope_bake',identity(signature)),None)
        if record is not None:_event(record,'shutdown')


    def scope_validate_payload(ticket, payload, provenance=None):
        """Validate the numerical envelope before admission is attempted.

        Only envelope defects relative to its capture ticket raise. A valid
        envelope whose ticket was superseded remains a normal stale result, so
        the coordinator can reject it without showing a false error.
        """
        if not isinstance(payload, Mapping):
            raise ScopePayloadError('Scope worker returned no numerical envelope')
        if provenance is None:
            try:provenance=payload['_scope_provenance']
            except (KeyError,TypeError):
                raise ScopePayloadError('Scope envelope has no captured provenance')
        try:provenance=tuple(tuple(item) for item in provenance)
        except (TypeError,ValueError):
            raise ScopePayloadError('Scope envelope provenance is malformed')
        analytical = bool(ticket.evaluation) and ticket.evaluation[0] != 'stack-final-lut'
        if not ((analytical and not provenance) or _final_provenance(provenance)):
            raise ScopePayloadError('Scope envelope is not a final 65³ capture')
        if ticket.resolution != 65 or provenance != ticket.provenance:
            raise ScopePayloadError('Scope envelope does not match its frozen capture')
        if ticket.evaluation:
            try: evaluation = tuple(payload['_scope_evaluation'])
            except (KeyError, TypeError, ValueError):
                raise ScopePayloadError('Scope envelope has no frozen evaluation identity')
            if evaluation != ticket.evaluation:
                raise ScopePayloadError('Scope envelope evaluator identity does not match its ticket')
        return provenance


    def scope_publish(ticket, payload=None, provenance=None):
        """Commit the immutable display envelope and admission state together."""
        if not _ticket_current(ticket):return False
        if not lifecycle_admits('publish'):return False
        record=_records.get(ticket.channel)
        if record is None or ticket.session != _session or payload is None:return False
        try:provenance=scope_validate_payload(ticket,payload,provenance)
        except ScopePayloadError:return False
        effects=record['store'].dispatch(policy.Event('publish',
            owner=ticket.owner,session=ticket.session,key=ticket.key,
            signal=ticket.signal,quality=ticket.quality,resolution=65,
            provenance=ticket.provenance_token,epoch=ticket.epoch,history=ticket.history))
        if not effects.accept:return False
        # Main-thread assignments cannot be interrupted by a Blender callback.
        # No external mutation or callbacks occur between payload and state.
        record.update(result=payload,display_ticket=ticket,error=None)
        record.pop('scope_previous',None)
        if ticket.display:
            display = _application.component('scopes.display')
            display.publish(record,ticket,payload,identity)
        return True


    def scope_result(node, kind='full', channel=None):
        if channel is None:channel = _application.component('scopes.sampling').channel(node)
        record=_records.get((kind,channel))
        if record is not None and 'scope_slots' in record:
            display = _application.component('scopes.display')
            result = display.result(record)
            return result if result is not None else record.get('scope_previous')
        return None if record is None else record.get('result')


    def scope_status(node, kind='full', channel=None):
        if channel is None:channel = _application.component('scopes.sampling').channel(node)
        record=_records.get((kind,channel))
        if record is None:return 'empty'
        state=record['store'].state
        if state.error:return 'error'
        if 'scope_slots' in record:
            display = _application.component('scopes.display')
            if display.result(record) is None:return 'stale' if record.get('scope_previous') is not None else 'empty'
            return 'current' if display.current(record) else 'stale'
        if not state.display_available:return 'empty'
        return ('current' if state.display_key == state.requested
                and state.display_signal == state.signal
                and state.display_epoch == state.epoch else 'stale')


    def scope_failed(ticket, error):
        """A failed task releases its slot while preserving the last good scope."""
        if not _ticket_current(ticket):return False
        scope_complete(ticket)
        record=_records.get(ticket.channel)
        if record is None or ticket.session != _session:return False
        effects=record['store'].dispatch(policy.Event('fail',
            owner=ticket.owner,session=ticket.session,key=ticket.key))
        if effects.failed:record.update(error=diagnostic_text(error))
        return bool(effects.failed)


    def effect_begin(ticket, phase):
        """Reserve one synchronous adapter effect before touching resources."""
        if not _ticket_current(ticket):return None
        if phase not in EFFECT_PHASES:raise ValueError('Unknown scheduler effect: '+phase)
        if not lifecycle_admits('timer' if phase == 'timer' else 'effect'):return None
        record=_records.get(ticket.channel)
        if record is None or ticket.session != _session:return None
        token=next(_serial);phase_id=EFFECT_PHASES.index(phase)+1
        effects=record['store'].dispatch(policy.Event('effect_begin',
            owner=ticket.owner,session=ticket.session,token=token,phase=phase_id))
        if not effects.effect:return None
        fields=dict(owner=ticket.owner, session=ticket.session, token=token, phase=phase_id)
        return EffectLease(ticket.channel,ticket.owner,ticket.session,token,phase_id,
            record['store'].event(policy.Event('effect_commit', **fields)),
            record['store'].event(policy.Event('effect_fail', **fields)))


    def _resource_ticket(kind, value):
        """Create a non-RNA lifetime for timers and other coordinator resources."""
        key=identity((kind,repr(value)));channel=(kind,key)
        record=_records.get(channel)
        if record is None:
            record=dict(store=_channel_store(),
                        context=(kind,key),node=None)
            _records[channel]=record
        _event(record,'request',key=key)
        _event(record,'signal',signal=key)
        return Ticket(channel,record['store'].state.owner,key,key,2,next(_serial),_session,
                      0,(),0,record['store'].state.epoch, application=_application.app.identity)


    def timer_effect(callback, phase, action, rollback=None):
        return effect(_resource_ticket('timer',callback),phase,action,rollback)


    def effect(ticket, phase, action, rollback=None):
        """Execute one admitted resource phase, reporting success or failure.

        A mutating caller supplies rollback of its staged resources. Rollback runs
        before the failure receipt, so no successful policy commit can describe a
        partial graph update. An interrupted rollback invalidates the lifetime.
        """
        lease=effect_begin(ticket,phase)
        if lease is None:raise RuntimeError('Scheduling effect belongs to a stale or occupied lifetime')
        try:
            result=action()
        except BaseException as exc:
            try:
                if rollback is not None:rollback()
            except BaseException:
                record=_records.get(ticket.channel)
                if record is not None:
                    _event(record,'invalidate')
                    _records.pop(ticket.channel,None)
                raise
            finally:
                finish_effect(lease,failed=True)
                record=_records.get(ticket.channel)
                if record is not None and record['store'].state.scope:scope_failed(ticket,exc)
            raise
        if not finish_effect(lease):
            if rollback is not None:rollback()
            raise RuntimeError('Scheduling effect invalidated before commit')
        return result


    def finish_effect(lease, failed=False):
        if type(lease) is not EffectLease:return False
        record = _records.get(lease.channel)
        if record is None or lease.session != _session:return False
        store = record['store']
        receipt = lease.failure if failed else lease.success
        # Reentry may replace this channel while the native action is running.
        # Report stale completion to effect() so it performs the staged rollback.
        if (receipt.owner != _application.app.identity or store.app is not _application.app
                or receipt.store != store.identity):
            return False
        effects = store.dispatch(receipt)
        return bool(effects.effect)


    def recover():
        """Fence every old callback after load/reload/crash recovery."""
        nonlocal _session
        for record in tuple(_records.values()):
            try:_event(record, 'recover')
            except (ReferenceError,RuntimeError):pass
        _session += 1
        clear_compositor()
        _records.clear();_gestures.clear();_keys.clear()
        return _session


    def scope_clear(kind=None):
        for channel in tuple(_records):
            if channel[0] in ({kind} if kind else {'linear', 'full'}):
                _event(_records.pop(channel), 'shutdown')


    def scope_forget(kind, pointer):
        """Eviction releases both worker admission and retained display slots."""
        record=_records.pop((kind,pointer),None)
        if record is not None:_event(record,'shutdown')


    def _pointer(value):
        try:
            return value.as_pointer() or None
        except (AttributeError, ReferenceError, RuntimeError):
            return None


    def _owner_pointer(value):
        """Return a stable output owner pointer for RNA nodes and Stack targets."""
        pointer = _pointer(value)
        try:
            stack = getattr(value, 'stack', None)
        except (AttributeError, ReferenceError, RuntimeError):
            stack = None
        owner = _pointer(stack) if stack is not None else pointer
        return owner


    def invalidate(node):
        """Retire all scheduler state owned by a deleted RNA node.

        This is deliberately pointer based: Blender can leave a Python wrapper
        alive after ``Node.free`` and may later reuse its numeric pointer. Every
        ticket and gesture attached to the old lifetime is rejected before that
        pointer can be assigned to a new node.
        """
        editing = _application.component('graph.editing')
        pointer = _pointer(node)
        if pointer is None:
            return
        native = not editing.is_target(node)
        def matches(target):
            return _pointer(target) == pointer or (native and _owner_pointer(target) == pointer)
        for channel, record in tuple(_records.items()):
            target = resolve(record.get('node'))
            if matches(target):
                _event(record, 'shutdown')
                _records.pop(channel, None)
        for token, edited_ref in tuple(_gestures.items()):
            edited = resolve(edited_ref)
            if matches(edited):
                _gestures.pop(token, None)
        for key, pending in tuple(_updates.items()):
            target = resolve(pending[0]) if isinstance(pending, tuple) else pending
            if matches(target):
                _updates.pop(key, None)
        for mapping in (_commits, _displayed, _preview_turn):
            mapping.pop(pointer, None)
        for signals in _frame_signals.values():
            signals.pop(pointer, None)
        for record in _records.values():
            if not _record_dragging(record):
                _event(record, 'release')


    def prune(live):
        """Invalidate records whose RNA owner disappeared since the last scan.

        ``live`` is produced once by ``nodes.refresh_all``. No Blender data scan
        occurs here; virtual Stack targets also validate their selected entry.
        """
        pointers = {_pointer(node) for node in live}
        pointers.discard(None)
        editing = _application.component('graph.editing')
        def valid(target):
            if editing.is_target(target):
                return _owner_pointer(target) in pointers and editing.valid(target)
            return _pointer(target) in pointers
        for channel, record in tuple(_records.items()):
            target = resolve(record.get('node'))
            if target is not None and not valid(target):
                _event(record, 'shutdown')
                _records.pop(channel, None)
        for token, edited_ref in tuple(_gestures.items()):
            edited = resolve(edited_ref)
            if not valid(edited):
                _gestures.pop(token, None)
        for key, (target_ref, _force) in tuple(_updates.items()):
            target = resolve(target_ref)
            if not valid(target):
                _updates.pop(key, None)
        for mapping in (_commits, _displayed, _preview_turn, *_frame_signals.values()):
            for key in tuple(mapping):
                if key not in pointers:
                    mapping.pop(key, None)
        for record in _records.values():
            if not _record_dragging(record):
                _event(record, 'release')


    def clear():
        """Invalidate policies while retaining application-timer registrations."""
        nonlocal _session
        for record in _records.values():
            _event(record, 'shutdown')
        _records.clear()
        _gestures.clear()
        _keys.clear()
        _session += 1


    def preview_running():
        import bpy
        return bool(_frames) or (not bpy.app.background and bpy.app.is_job_running('COMPOSITE'))


    def can_commit(node):
        import bpy
        import time
        output_guard = _application.component('graph.output_guard')
        if not output_guard.available():return False
        if node.id_data.bl_idname != 'CompositorNodeTree':return True
        compositor = _application.component('graph.compositor')
        if compositor.preview_running():return False
        return bpy.app.background or time.monotonic() >= _preview_turn.get(node.as_pointer(),0.)


    def preview_installed(node):
        import time
        if node.id_data.bl_idname == 'CompositorNodeTree':
            _preview_turn[node.as_pointer()] = time.monotonic() + .016


    def frame_start(scene):
        import bpy
        pointer=scene.as_pointer();_frames.add(pointer)
        root=scene.compositing_node_group
        owners={tree.as_pointer() for tree in bpy.data.node_groups
                if root and (tree==root or root.contains_tree(tree))}
        _frame_signals[pointer]={key:token for key,(owner,token) in _commits.items() if owner in owners}


    def frame_end(scene, cancelled=False):
        pointer=scene.as_pointer();_frames.discard(pointer)
        signals=_frame_signals.pop(pointer,{})
        if not cancelled:_displayed.update(signals)


    def defer_update(node, force=False):
        compositor = _application.component('graph.compositor')
        if compositor.can_commit(node):return False
        key=node.as_pointer();old=_updates.get(key)
        _updates[key]=(retain(node,'scheduler.update'),force or bool(old and old[1]))
        wake(compositor.update_tick,.016)
        return True


    def flush_update(node):
        pending=_updates.pop(node.as_pointer(),None)
        if pending:
            nodes = _application.component('graph.nodes'); stacks = _application.component('graph.stacks')
            if stacks.is_stack(node):stacks.refresh(node,force=pending[1],immediate=True)
            else:nodes.refresh_node(node,force=pending[1],immediate=True)


    def update_tick():
        import bpy
        jobs = _application.component('scopes.jobs')
        compositor = _application.component('graph.compositor'); nodes = _application.component('graph.nodes'); stacks = _application.component('graph.stacks'); lut = _application.component('graph.lut')
        jobs.linear_tick()
        if not _updates:return None
        if compositor.preview_running() or bpy.app.is_job_running('RENDER'):return .016
        live={node.as_pointer():node for node in nodes.all_nodes()}
        for pointer,(node,force) in tuple(_updates.items()):
            if pointer not in live:_updates.pop(pointer,None);continue
            node=live[pointer]
            if not compositor.can_commit(node):continue
            state=lut._states.get(pointer)
            if (state is not None and not state['final_published']
                    and state['baker'].preview is not None and not state['preview_published']):
                continue
            _updates.pop(pointer,None)
            try:
                if stacks.is_stack(node):stacks.refresh(node,force=force,immediate=True)
                else:nodes.refresh_node(node,force=force,immediate=True)
            except (ReferenceError,RuntimeError):log_exception(compositor.LOG,'Perceptual Grade compositor update failed')
        return .016 if _updates else None


    def clear_compositor():
        _updates.clear();_frames.clear();_frame_signals.clear()
        _commits.clear();_displayed.clear();_preview_turn.clear()


    def shutdown():
        """Release every registered timer and invalidate all scheduling lifetimes.

        No scene-data access is needed, so partial enable and restricted add-on
        registration can use the same cleanup as normal disable and reload.
        """
        for callback in tuple(_timers):
            unwake(callback)
        clear()
        clear_compositor()


    def timer_callback(callback):
        """The exact registered handle for a logical foreground callback."""
        if callback in _timer_boundaries.values():
            return callback
        if callback not in _timer_boundaries:
            from functools import wraps
            boundary = _application.component('integration.outcomes').callback(
                callback, control='timer.' + callback.__name__,
                recovery='perceptual_grade.retry_recovery')
            @wraps(boundary)
            def tick():
                result = boundary()
                if result is None:
                    _timers.discard(tick)
                    _timer_boundaries.pop(callback, None)
                return result
            _timer_boundaries[callback] = tick
        return _timer_boundaries[callback]


    def wake(callback, delay, persistent=False):
        if _application.closing:
            raise RuntimeError('Cannot schedule new callbacks while application cleanup is pending')
        import bpy
        callback = timer_callback(callback)
        if not bpy.app.timers.is_registered(callback):
            # Track intent before registration so partial failure cannot orphan a
            # callback. If rollback itself fails, shutdown still has its handle.
            _timers.add(callback)
            def rollback():
                if bpy.app.timers.is_registered(callback):bpy.app.timers.unregister(callback)
                _timers.discard(callback)
            timer_effect(callback,'timer',lambda:bpy.app.timers.register(
                callback, first_interval=delay, persistent=persistent),rollback=rollback)
        _timers.add(callback)


    def unwake(callback):
        import bpy
        callback = _timer_boundaries.get(callback, callback)
        if bpy.app.timers.is_registered(callback):
            timer_effect(callback,'timer',lambda:bpy.app.timers.unregister(callback))
        _timers.discard(callback)
        for original, native in tuple(_timer_boundaries.items()):
            if native is callback:
                _timer_boundaries.pop(original)

    def _set_export_EFFECT_PHASES(value):
        nonlocal EFFECT_PHASES
        EFFECT_PHASES = value

    def _set_export__commits(value):
        nonlocal _commits
        _commits = value

    def _set_export__displayed(value):
        nonlocal _displayed
        _displayed = value

    def _set_export__frame_signals(value):
        nonlocal _frame_signals
        _frame_signals = value

    def _set_export__frames(value):
        nonlocal _frames
        _frames = value

    def _set_export__gestures(value):
        nonlocal _gestures
        _gestures = value

    def _set_export__keys(value):
        nonlocal _keys
        _keys = value

    def _set_export__lifecycle_store(value):
        nonlocal _lifecycle_store
        _lifecycle_store = value

    def _set_export__preview_turn(value):
        nonlocal _preview_turn
        _preview_turn = value

    def _set_export__records(value):
        nonlocal _records
        _records = value

    def _set_export__serial(value):
        nonlocal _serial
        _serial = value

    def _set_export__session(value):
        nonlocal _session
        _session = value

    def _set_export__timers(value):
        nonlocal _timers
        _timers = value

    def _set_export__updates(value):
        nonlocal _updates
        _updates = value

    def _set_export_Ticket(value):
        nonlocal Ticket
        Ticket = value

    def _set_export_ScopePayloadError(value):
        nonlocal ScopePayloadError
        ScopePayloadError = value

    def _set_export_EffectLease(value):
        nonlocal EffectLease
        EffectLease = value

    def _set_export_session(value):
        nonlocal session
        session = value

    def _set_export_lifecycle_event(value):
        nonlocal lifecycle_event
        lifecycle_event = value

    def _set_export_lifecycle_admits(value):
        nonlocal lifecycle_admits
        lifecycle_admits = value

    def _set_export_identity(value):
        nonlocal identity
        identity = value

    def _set_export_output(value):
        nonlocal output
        output = value

    def _set_export__affected(value):
        nonlocal _affected
        _affected = value

    def _set_export__scope_affected(value):
        nonlocal _scope_affected
        _scope_affected = value

    def _set_export__gesture_matches(value):
        nonlocal _gesture_matches
        _gesture_matches = value

    def _set_export_dragging(value):
        nonlocal dragging
        dragging = value

    def _set_export_scope_dragging(value):
        nonlocal scope_dragging
        scope_dragging = value

    def _set_export__record_dragging(value):
        nonlocal _record_dragging
        _record_dragging = value

    def _set_export__event(value):
        nonlocal _event
        _event = value

    def _set_export__channel_store(value):
        nonlocal _channel_store
        _channel_store = value

    def _set_export__ticket_current(value):
        nonlocal _ticket_current
        _ticket_current = value

    def _set_export__record(value):
        nonlocal _record
        _record = value

    def _set_export_begin(value):
        nonlocal begin
        begin = value

    def _set_export_end(value):
        nonlocal end
        end = value

    def _set_export_lut_invalidated(value):
        nonlocal lut_invalidated
        lut_invalidated = value

    def _set_export_lut_request(value):
        nonlocal lut_request
        lut_request = value

    def _set_export_lut_candidate(value):
        nonlocal lut_candidate
        lut_candidate = value

    def _set_export_lut_cache_hit(value):
        nonlocal lut_cache_hit
        lut_cache_hit = value

    def _set_export_publication(value):
        nonlocal publication
        publication = value

    def _set_export_ready(value):
        nonlocal ready
        ready = value

    def _set_export_replacement(value):
        nonlocal replacement
        replacement = value

    def _set_export_scope_request(value):
        nonlocal scope_request
        scope_request = value

    def _set_export_scope_coherent(value):
        nonlocal scope_coherent
        scope_coherent = value

    def _set_export__final_provenance(value):
        nonlocal _final_provenance
        _final_provenance = value

    def _set_export_scope_start(value):
        nonlocal scope_start
        scope_start = value

    def _set_export_scope_retry(value):
        nonlocal scope_retry
        scope_retry = value

    def _set_export_scope_complete(value):
        nonlocal scope_complete
        scope_complete = value

    def _set_export_scope_bake_request(value):
        nonlocal scope_bake_request
        scope_bake_request = value

    def _set_export_scope_bake_clear(value):
        nonlocal scope_bake_clear
        scope_bake_clear = value

    def _set_export_scope_validate_payload(value):
        nonlocal scope_validate_payload
        scope_validate_payload = value

    def _set_export_scope_publish(value):
        nonlocal scope_publish
        scope_publish = value

    def _set_export_scope_result(value):
        nonlocal scope_result
        scope_result = value

    def _set_export_scope_status(value):
        nonlocal scope_status
        scope_status = value

    def _set_export_scope_failed(value):
        nonlocal scope_failed
        scope_failed = value

    def _set_export_effect_begin(value):
        nonlocal effect_begin
        effect_begin = value

    def _set_export__resource_ticket(value):
        nonlocal _resource_ticket
        _resource_ticket = value

    def _set_export_timer_effect(value):
        nonlocal timer_effect
        timer_effect = value

    def _set_export_effect(value):
        nonlocal effect
        effect = value

    def _set_export_finish_effect(value):
        nonlocal finish_effect
        finish_effect = value

    def _set_export_recover(value):
        nonlocal recover
        recover = value

    def _set_export_scope_clear(value):
        nonlocal scope_clear
        scope_clear = value

    def _set_export_scope_forget(value):
        nonlocal scope_forget
        scope_forget = value

    def _set_export__pointer(value):
        nonlocal _pointer
        _pointer = value

    def _set_export__owner_pointer(value):
        nonlocal _owner_pointer
        _owner_pointer = value

    def _set_export_invalidate(value):
        nonlocal invalidate
        invalidate = value

    def _set_export_prune(value):
        nonlocal prune
        prune = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_preview_running(value):
        nonlocal preview_running
        preview_running = value

    def _set_export_can_commit(value):
        nonlocal can_commit
        can_commit = value

    def _set_export_preview_installed(value):
        nonlocal preview_installed
        preview_installed = value

    def _set_export_frame_start(value):
        nonlocal frame_start
        frame_start = value

    def _set_export_frame_end(value):
        nonlocal frame_end
        frame_end = value

    def _set_export_defer_update(value):
        nonlocal defer_update
        defer_update = value

    def _set_export_flush_update(value):
        nonlocal flush_update
        flush_update = value

    def _set_export_update_tick(value):
        nonlocal update_tick
        update_tick = value

    def _set_export_clear_compositor(value):
        nonlocal clear_compositor
        clear_compositor = value

    def _set_export_shutdown(value):
        nonlocal shutdown
        shutdown = value

    def _set_export_wake(value):
        nonlocal wake
        wake = value

    def _set_export_unwake(value):
        nonlocal unwake
        unwake = value

    return Component(readers={
        "__name__": lambda: __name__,
        'EFFECT_PHASES': lambda: EFFECT_PHASES,
        'EffectLease': lambda: EffectLease,
        'Mapping': lambda: Mapping,
        'OrderedDict': lambda: OrderedDict,
        'ScopePayloadError': lambda: ScopePayloadError,
        'Ticket': lambda: Ticket,
        '_affected': lambda: _affected,
        '_channel_store': lambda: _channel_store,
        '_commits': lambda: _commits,
        '_displayed': lambda: _displayed,
        '_event': lambda: _event,
        '_final_provenance': lambda: _final_provenance,
        '_frame_signals': lambda: _frame_signals,
        '_frames': lambda: _frames,
        '_gesture_matches': lambda: _gesture_matches,
        '_gestures': lambda: _gestures,
        '_keys': lambda: _keys,
        '_lifecycle_store': lambda: _lifecycle_store,
        '_owner_pointer': lambda: _owner_pointer,
        '_pointer': lambda: _pointer,
        '_preview_turn': lambda: _preview_turn,
        '_record': lambda: _record,
        '_record_dragging': lambda: _record_dragging,
        '_records': lambda: _records,
        '_resource_ticket': lambda: _resource_ticket,
        '_scope_affected': lambda: _scope_affected,
        '_serial': lambda: _serial,
        '_session': lambda: _session,
        '_ticket_current': lambda: _ticket_current,
        '_timers': lambda: _timers,
        '_updates': lambda: _updates,
        'begin': lambda: begin,
        'can_commit': lambda: can_commit,
        'clear': lambda: clear,
        'clear_compositor': lambda: clear_compositor,
        'count': lambda: count,
        'dataclass': lambda: dataclass,
        'defer_update': lambda: defer_update,
        'diagnostic_text': lambda: diagnostic_text,
        'dragging': lambda: dragging,
        'effect': lambda: effect,
        'effect_begin': lambda: effect_begin,
        'end': lambda: end,
        'finish_effect': lambda: finish_effect,
        'flush_update': lambda: flush_update,
        'frame_end': lambda: frame_end,
        'frame_start': lambda: frame_start,
        'identity': lambda: identity,
        'invalidate': lambda: invalidate,
        'lifecycle_admits': lambda: lifecycle_admits,
        'lifecycle_event': lambda: lifecycle_event,
        'log_exception': lambda: log_exception,
        'lut_cache_hit': lambda: lut_cache_hit,
        'lut_candidate': lambda: lut_candidate,
        'lut_invalidated': lambda: lut_invalidated,
        'lut_request': lambda: lut_request,
        'output': lambda: output,
        'policy': lambda: policy,
        'preview_installed': lambda: preview_installed,
        'preview_running': lambda: preview_running,
        'prune': lambda: prune,
        'publication': lambda: publication,
        'ready': lambda: ready,
        'recover': lambda: recover,
        'replacement': lambda: replacement,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'scope_bake_clear': lambda: scope_bake_clear,
        'scope_bake_request': lambda: scope_bake_request,
        'scope_clear': lambda: scope_clear,
        'scope_coherent': lambda: scope_coherent,
        'scope_complete': lambda: scope_complete,
        'scope_dragging': lambda: scope_dragging,
        'scope_failed': lambda: scope_failed,
        'scope_forget': lambda: scope_forget,
        'scope_publish': lambda: scope_publish,
        'scope_request': lambda: scope_request,
        'scope_result': lambda: scope_result,
        'scope_retry': lambda: scope_retry,
        'scope_start': lambda: scope_start,
        'scope_status': lambda: scope_status,
        'scope_validate_payload': lambda: scope_validate_payload,
        'session': lambda: session,
        'shutdown': lambda: shutdown,
        'timer_effect': lambda: timer_effect,
        'timer_callback': lambda: timer_callback,
        'unwake': lambda: unwake,
        'update_tick': lambda: update_tick,
        'uuid4': lambda: uuid4,
        'wake': lambda: wake,
    }, writers={
        'EFFECT_PHASES': _set_export_EFFECT_PHASES,
        '_commits': _set_export__commits,
        '_displayed': _set_export__displayed,
        '_frame_signals': _set_export__frame_signals,
        '_frames': _set_export__frames,
        '_gestures': _set_export__gestures,
        '_keys': _set_export__keys,
        '_lifecycle_store': _set_export__lifecycle_store,
        '_preview_turn': _set_export__preview_turn,
        '_records': _set_export__records,
        '_serial': _set_export__serial,
        '_session': _set_export__session,
        '_timers': _set_export__timers,
        '_updates': _set_export__updates,
    'Ticket': _set_export_Ticket, 'ScopePayloadError': _set_export_ScopePayloadError, 'EffectLease': _set_export_EffectLease, 'session': _set_export_session, 'lifecycle_event': _set_export_lifecycle_event, 'lifecycle_admits': _set_export_lifecycle_admits, 'identity': _set_export_identity, 'output': _set_export_output, '_affected': _set_export__affected, '_scope_affected': _set_export__scope_affected, '_gesture_matches': _set_export__gesture_matches, 'dragging': _set_export_dragging, 'scope_dragging': _set_export_scope_dragging, '_record_dragging': _set_export__record_dragging, '_event': _set_export__event, '_channel_store': _set_export__channel_store, '_ticket_current': _set_export__ticket_current, '_record': _set_export__record, 'begin': _set_export_begin, 'end': _set_export_end, 'lut_invalidated': _set_export_lut_invalidated, 'lut_request': _set_export_lut_request, 'lut_candidate': _set_export_lut_candidate, 'lut_cache_hit': _set_export_lut_cache_hit, 'publication': _set_export_publication, 'ready': _set_export_ready, 'replacement': _set_export_replacement, 'scope_request': _set_export_scope_request, 'scope_coherent': _set_export_scope_coherent, '_final_provenance': _set_export__final_provenance, 'scope_start': _set_export_scope_start, 'scope_retry': _set_export_scope_retry, 'scope_complete': _set_export_scope_complete, 'scope_bake_request': _set_export_scope_bake_request, 'scope_bake_clear': _set_export_scope_bake_clear, 'scope_validate_payload': _set_export_scope_validate_payload, 'scope_publish': _set_export_scope_publish, 'scope_result': _set_export_scope_result, 'scope_status': _set_export_scope_status, 'scope_failed': _set_export_scope_failed, 'effect_begin': _set_export_effect_begin, '_resource_ticket': _set_export__resource_ticket, 'timer_effect': _set_export_timer_effect, 'effect': _set_export_effect, 'finish_effect': _set_export_finish_effect, 'recover': _set_export_recover, 'scope_clear': _set_export_scope_clear, 'scope_forget': _set_export_scope_forget, '_pointer': _set_export__pointer, '_owner_pointer': _set_export__owner_pointer, 'invalidate': _set_export_invalidate, 'prune': _set_export_prune, 'clear': _set_export_clear, 'preview_running': _set_export_preview_running, 'can_commit': _set_export_can_commit, 'preview_installed': _set_export_preview_installed, 'frame_start': _set_export_frame_start, 'frame_end': _set_export_frame_end, 'defer_update': _set_export_defer_update, 'flush_update': _set_export_flush_update, 'update_tick': _set_export_update_tick, 'clear_compositor': _set_export_clear_compositor, 'shutdown': _set_export_shutdown, 'wake': _set_export_wake, 'unwake': _set_export_unwake})
