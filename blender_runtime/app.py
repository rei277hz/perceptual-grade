"""Explicit application ownership, machine Stores and terminal fencing."""
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary, finalize

from .machine import DomainError, Machine
from ..invariant import AdmissionError, ActorFailed, ActorSystem
from ..invariant.machine import validate


class ForeignOwnerError(ValueError):
    """A value or capability belongs to a different App lifecycle."""


class ClosedError(RuntimeError):
    """A fenced App or Store cannot admit new work."""


class PendingTransitionError(TimeoutError):
    """A queued model operation has an unresolved outcome; never resubmit it."""
    def __init__(self, token, operation):
        self.token, self.operation = token, operation
        super().__init__(f'Actor operation {operation} is pending ({token}); reconcile its receipt')


@dataclass(frozen=True, slots=True)
class Envelope:
    owner: str
    store: int
    value: object


@dataclass(frozen=True, slots=True)
class Transition:
    state: object
    effects: object


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Factory receives its App; storing it does not construct a runtime."""
    factory: object

    def __post_init__(self):
        if not callable(self.factory):
            raise TypeError('Execution configuration requires a runtime factory')


class ExecutionService:
    def __init__(self, app, config):
        self.app, self.config, self._runtime = app, config, None

    def get_runtime(self):
        self.app.require_open()
        if self._runtime is None:
            runtime = self.config.factory(self.app)
            if getattr(runtime, 'closed', False):
                raise ClosedError('Execution factory returned a closed runtime')
            owner = getattr(runtime, 'app_identity', None)
            if owner not in (None, self.app.identity):
                raise ForeignOwnerError('Execution runtime belongs to another App')
            runtime.app_identity = self.app.identity
            self._runtime = runtime
        elif self._runtime.closed:
            raise ClosedError('A closed runtime requires a fresh App lifecycle')
        return self._runtime

    def current_runtime(self):
        return self._runtime

    def fence(self):
        if self._runtime is not None:
            self._runtime.fenced = True

    def close(self):
        if self._runtime is not None:
            report = self._runtime.shutdown()
            if getattr(report, 'clean', True) is not True:
                terminal = getattr(self._runtime, 'physical_terminal', None)
                if not callable(terminal) or terminal() is not True:
                    raise RuntimeError('Execution cleanup did not reach its physical terminal boundary')


class ActorHost:
    """One private actor scheduler; all native effects stay on the host thread."""
    def __init__(self, app):
        import queue
        import threading
        self.app = app
        self.timeout = 10.0
        self.native = queue.Queue()
        self.faults = queue.Queue()
        self.pending = {}
        self.reconciled = {}
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name='blender-runtime-actors', daemon=True)
        self.thread.start()
        if not self._ready.wait(self.timeout):
            raise RuntimeError('actor scheduler did not start')
        if hasattr(self, 'startup_error'):
            raise RuntimeError('actor scheduler failed') from self.startup_error

    def _run(self):
        import asyncio
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self.startup_error = error
            self._ready.set()
            self.faults.put(error)

    async def _serve(self):
        import asyncio
        self.loop = asyncio.get_running_loop()
        self.system = ActorSystem(on_fault=self.faults.put)
        self._stop = asyncio.Event()
        self._ready.set()
        await self._stop.wait()
        await self.system.close()

    def submit(self, coroutine):
        import asyncio
        import threading
        if threading.current_thread() is self.thread or self.loop.is_closed():
            coroutine.close()
            raise ClosedError('host bridge cannot block its actor scheduler')
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def wait(self, future, operation):
        from concurrent.futures import TimeoutError as FutureTimeout
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            if future.done():
                self.app._fence_admission()
                raise
            # Do not cancel the future: cancellation does not roll back a
            # queued or committed transition. Retain its exact eventual result.
            token = uuid4().hex
            self.pending[token] = (operation, future)
            self.app._fence_admission()
            raise PendingTransitionError(token, operation) from None
        except AdmissionError:
            raise
        except Exception:
            self.app._fence_admission()
            raise

    def call(self, coroutine, operation):
        return self.wait(self.submit(coroutine), operation)

    def reconcile(self, token):
        from concurrent.futures import TimeoutError as FutureTimeout
        if token in self.reconciled:
            result = self.reconciled[token]
        else:
            operation, future = self.pending[token]
            try:
                result = dict(operation=operation, result=future.result(timeout=self.timeout), error=None)
            except FutureTimeout:
                raise PendingTransitionError(token, operation) from None
            except Exception as error:
                result = dict(operation=operation, result=None, error=error)
            self.pending.pop(token)
            self.reconciled[token] = result
        if result['error'] is not None:
            raise result['error']
        return result['result']

    def drain_pending(self):
        for token in tuple(self.pending):
            self.reconcile(token)

    def spawn(self, name, machine, arguments, initializer):
        async def create():
            ref = self.system.spawn(name, machine, arguments=arguments, initializer=initializer)
            await ref.ready()
            return ref
        return self.call(create(), 'spawn:' + name)

    def close(self):
        self.drain_pending()
        if not self.thread.is_alive():
            return
        self.loop.call_soon_threadsafe(self._stop.set)
        self.thread.join(timeout=self.timeout)
        if self.thread.is_alive():
            raise RuntimeError('actor scheduler still has outstanding retirement work')


class Store:
    """Host capability for a shared ActorRef; state exists only in its task."""
    def __init__(self, app, identity, machine, initial, initializer):
        import inspect
        self.app, self.identity, self.machine = app, identity, machine
        name = machine.default_initializer if initializer is None else initializer
        if name not in machine.initializers:
            raise DomainError('Unknown checked initializer: ' + str(name))
        if type(initial) is not dict or set(initial) != set(inspect.signature(machine._functions[name]).parameters):
            raise AdmissionError('initializer arguments must match their complete declaration')
        self.ref = app.actors.spawn(f'{machine.module.__name__}:{identity}', machine, initial, name)
        self._finalizer = finalize(self, self.ref.close_threadsafe)

    @property
    def state(self):
        return self.app.actors.call(self.ref.snapshot(), 'snapshot:' + str(self.identity))

    def event(self, event):
        """Bind a receipt/event to this exact Store and App before native work."""
        self.app.require_open()
        validate(event, self.machine.event_type, 'event')
        return Envelope(self.app.identity, self.identity, event)

    def _unwrap(self, event):
        if isinstance(event, Envelope):
            if event.owner != self.app.identity or event.store != self.identity:
                raise ForeignOwnerError('Event/receipt belongs to another App or Store')
            event = event.value
        return event

    def preview(self, event):
        self.app.require_open()
        state, effects = self.app.actors.call(self.ref.preview(self._unwrap(event)), 'preview:' + str(self.identity))
        return Transition(state, effects)

    def dispatch(self, event):
        self.app.require_open()
        return self._dispatch_cleanup(event)

    def _dispatch_cleanup(self, event):
        """Framework-owned finalizers may retire resources after admission fencing."""
        return self.app.actors.call(self.ref.dispatch(self._unwrap(event)), 'dispatch:' + str(self.identity))

    def query(self, name, **parameters):
        self.app.require_open()
        return self._query_cleanup(name, **parameters)

    def _query_cleanup(self, name, **parameters):
        import inspect
        if name not in self.machine.queries:
            raise DomainError('Unknown machine query: ' + str(name))
        if 'state' not in inspect.signature(self.machine._functions[name]).parameters:
            try:
                return self.machine.call(name, **parameters)
            except AdmissionError:
                raise
            except Exception:
                self.app._fence_admission()
                raise
        return self.app.actors.call(self.ref.query(name, **parameters), 'query:' + str(self.identity))

    def close(self):
        self.app.actors.call(self.ref.close(), 'close:' + str(self.identity))
        self._finalizer.detach()


class App:
    """One explicit lifecycle, with optional lazy execution and scoped resources.

    Products perform their native recovery protocol before close. This class
    provides terminal fencing and owned cleanup, not product rollback policy.
    """
    def __init__(self, *, package, storage_name, execution=None, ownership=None, blender=None, lifecycle=None):
        import queue
        if type(storage_name) is not str or not storage_name.isidentifier():
            raise ValueError('Storage name must be an extension identifier')
        if not isinstance(package, (str, Path)) or not str(package):
            raise ValueError('App requires its explicit package identity/path')
        self.package, self.storage_name = package, storage_name
        self.identity = uuid4().hex
        self.fenced = self.closed = False
        self._actors = None
        self._stores = WeakValueDictionary()
        self._next_store = 1
        self._retirement_callbacks = []
        self._finalizers = queue.Queue()
        self._draining_finalizers = False
        self.lifecycle = None
        if execution is not None and type(execution) is not ExecutionConfig:
            raise TypeError('Execution must be an ExecutionConfig')
        self.execution = None if execution is None else ExecutionService(self, execution)
        self.ownership = None
        self.blender = None
        if ownership is not None:
            from .ownership import OwnershipConfig, OwnershipContext
            if type(ownership) is not OwnershipConfig:
                raise TypeError('Ownership must be an OwnershipConfig')
            self.ownership = OwnershipContext(self, ownership)
        if blender is not None:
            from .blender import BlenderConfig, BlenderContext
            if type(blender) is not BlenderConfig:
                raise TypeError('Blender must be a BlenderConfig')
            self.blender = BlenderContext(self, blender)
        if lifecycle is not None:
            from .protocol import Protocol
            self.lifecycle = Protocol(self, lifecycle)

    def require_open(self):
        if self.fenced or self.closed:
            raise ClosedError('App is terminally fenced')

    def on_retired(self, callback):
        """Run product retirement after helper exit, while model actors live."""
        self.require_open()
        if not callable(callback):
            raise TypeError('retirement hook must be callable')
        if callback not in self._retirement_callbacks:
            self._retirement_callbacks.append(callback)

    def _defer_finalizer(self, callback, *values):
        """Framework cleanup carries only weak owner references and scalar IDs."""
        import threading
        if threading.current_thread() is threading.main_thread():
            callback(*values)
        else:
            self._finalizers.put((callback, values))

    def drain_finalizers(self):
        """Apply pending decrements only at an explicit foreground boundary."""
        import queue
        import threading
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('native finalizers require the host main thread')
        if self._draining_finalizers:
            return
        self._draining_finalizers = True
        try:
            while True:
                try:
                    callback, values = self._finalizers.get_nowait()
                except queue.Empty:
                    return
                try:
                    callback(*values)
                except Exception:
                    self._finalizers.put((callback, values))
                    self._fence_admission()
                    raise
        finally:
            self._draining_finalizers = False

    @property
    def actors(self):
        if self._actors is None:
            self.require_open()
            self._actors = ActorHost(self)
        return self._actors

    def reconcile(self, token):
        if self._actors is None:
            raise KeyError(token)
        return self._actors.reconcile(token)

    def pump_native(self, limit=32):
        import queue
        import threading
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('native effects require the host main thread')
        self.drain_finalizers()
        if self._actors is None:
            return
        for _ in range(limit):
            try:
                callback = self._actors.native.get_nowait()
            except queue.Empty:
                break
            callback()
        if not self._actors.faults.empty():
            fault = self._actors.faults.get_nowait()
            self.fence()
            raise RuntimeError('protocol actor failed; native admission is fenced') from getattr(fault, 'error', fault)

    def require_store(self, store):
        if not isinstance(store, Store) or store.app is not self or self._stores.get(store.identity) is not store:
            raise ForeignOwnerError('Store belongs to another App')
        return store

    def store(self, *, machine, initial=None, initializer=None):
        self.require_open()
        if type(machine) is not Machine:
            raise TypeError('App.store requires a Machine descriptor')
        identity = self._next_store
        self._next_store += 1
        store = Store(self, identity, machine, {} if initial is None else initial, initializer)
        self._stores[identity] = store
        return store

    def _fence_admission(self):
        self.fenced = True
        if self.execution is not None:
            self.execution.fence()

    def fence(self):
        self._fence_admission()
        if (self.lifecycle is not None and not self.closed
                and (self._actors is None or not self._actors.pending)
                and not self.lifecycle.state.closing):
            self.lifecycle.require('close', 'fence')

    def close(self):
        self.drain_finalizers()
        if self.closed:
            return
        self._fence_admission()
        if self._actors is not None:
            self._actors.drain_pending()
        self.fence()
        errors = []
        # Native producers retire first. Their product leases and actor-owned
        # receipts remain usable until the corresponding callbacks finish.
        if self.execution is not None:
            try:
                self.execution.close()
            except Exception as error:
                errors.append(error)
        if not errors:
            for callback in tuple(self._retirement_callbacks):
                try:
                    callback()
                except Exception as error:
                    errors.append(error)
                else:
                    self._retirement_callbacks.remove(callback)
        # Failed retirement retains the original ownership and registration
        # ledgers; a later close retries the exact remaining callbacks.
        if not errors and self.ownership is not None:
            try:
                self.ownership.close()
            except Exception as error:
                errors.append(error)
        if not errors and self.lifecycle is None and self.blender is not None:
            try:
                self.blender.close()
            except Exception as error:
                errors.append(error)
        if errors:
            if self.lifecycle is not None:
                self.lifecycle.event('failed')
            raise ExceptionGroup('App cleanup has not reached its terminal boundary', errors)
        if self.lifecycle is not None:
            if self.lifecycle.state.phase == 'repair_needed':
                self.lifecycle.require('repair', 'repair')
                self.lifecycle.require('repaired', 'fence', validated=True, debts=0)
            if self.lifecycle.state.phase == 'draining':
                self.lifecycle.require('drained', 'release', workers=0, borrows=0,
                                       readers=0, debts=0, validated=True)
            try:
                if self.blender is not None:
                    self.blender.close()
            except Exception:
                self.lifecycle.event('failed')
                raise
            if not self.lifecycle.event('released', validated=True).accepted:
                raise RuntimeError('lifecycle rejected native registration release receipt')
        if self._actors is not None:
            self._actors.close()
        self.drain_finalizers()
        self.closed = True

