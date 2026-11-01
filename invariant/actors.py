"""One task and inbox per automaton, with explicitly supplied effect outcomes."""

import asyncio
from concurrent.futures import Future, InvalidStateError
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import math
import threading

from .app import ClosedError
from .machine import AdmissionError, ContractError, Store, validate


def _admit(event, machine):
    try:
        validate(event, machine.event_type, 'event')
    except ContractError as error:
        raise AdmissionError(str(error)) from error


@dataclass(frozen=True)
class ActorFault:
    actor: object
    stage: str
    error: BaseException


class ActorFailed(RuntimeError):
    def __init__(self, fault):
        self.fault = fault
        super().__init__(f'actor {fault.actor.name!r} failed during {fault.stage}: {fault.error}')


@dataclass
class _Request:
    kind: str
    value: object = None
    reply: object = None
    recovery: bool = False


@dataclass(eq=False)
class _Actor:
    name: str
    machine: object
    initializer: str
    arguments: dict
    on_effects: object
    on_failure: object
    close_event: object
    is_closed: object
    inbox: object = field(default_factory=asyncio.Queue)
    initialized: object = field(default_factory=Future)
    stopped: object = field(default_factory=Future)
    effects: set = field(default_factory=set)
    ref: object = None
    task: object = None
    fault: object = None
    closing: bool = False
    model_closed: bool = False
    close_sent: bool = False
    close_task: object = None
    stopping: bool = False


def _answer(future, value=None, error=None):
    if future is not None and not future.done():
        try:
            if error is None:
                future.set_result(value)
            else:
                future.set_exception(error)
        except InvalidStateError:
            # A submitting thread can cancel its acknowledgment concurrently.
            pass


class ActorRef:
    """A mailbox address; no live Store or mutable state is exposed."""

    def __init__(self, system, actor):
        self._system, self._actor = system, actor

    @property
    def name(self):
        return self._actor.name

    @property
    def closed(self):
        return self._actor.stopped.done()

    def tell(self, event):
        """Submit without blocking, returning a thread-safe dispatch acknowledgment."""
        _admit(event, self._actor.machine)
        return self._system._submit(self._actor, _Request('event', event, Future()))

    async def dispatch(self, event):
        return await asyncio.wrap_future(self.tell(event))

    async def ready(self):
        await asyncio.shield(asyncio.wrap_future(self._actor.initialized))
        return self

    def snapshot_threadsafe(self):
        return self._system._submit(self._actor, _Request('snapshot', reply=Future()))

    async def snapshot(self):
        return await asyncio.wrap_future(self.snapshot_threadsafe())

    def preview_threadsafe(self, event):
        _admit(event, self._actor.machine)
        return self._system._submit(self._actor, _Request('preview', event, Future()))

    async def preview(self, event):
        return await asyncio.wrap_future(self.preview_threadsafe(event))

    def query_threadsafe(self, name, **arguments):
        return self._system._submit(self._actor, _Request('query', (name, deepcopy(arguments)), Future()))

    async def query(self, name, **arguments):
        return await asyncio.wrap_future(self.query_threadsafe(name, **arguments))

    def close_threadsafe(self):
        """Request this actor's declared shutdown and drain; never cancel its task."""
        return self._system._request_close(self._actor)

    async def close(self):
        await asyncio.wrap_future(self.close_threadsafe())


class ActorSystem:
    """Hosts actors on the current asyncio loop; physical work uses external executors.

    Effect routing is synchronous and brief. It can send peer messages, schedule
    timers, or register asynchronous work with run_effect; it must not block.
    Shutdown sends the declared close event and waits for closed model states,
    registered work, and queued receipts before stopping the actor loops.
    """

    def __init__(self, *, on_fault=None):
        self.loop = asyncio.get_running_loop()
        self._thread = threading.get_ident()
        self._actors = {}
        self._retirements = set()
        self._timers = {}
        self._changed = asyncio.Event()
        self._closing = False
        self._closed = False
        self._close_task = None
        self.on_fault = on_fault
        self.faults = []

    def _check_loop(self):
        if threading.get_ident() != self._thread or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError('actor system operation requires its owning event loop')

    def spawn(self, name, machine, *, arguments=None, initializer='new', on_effects=None,
              on_failure=None, close_event=None, is_closed=None):
        self._check_loop()
        if self._closing:
            raise ClosedError('actor system is closing')
        if not isinstance(name, str) or not name or name in self._actors:
            raise ValueError('actor names must be nonempty and unique')
        if (close_event is None) != (is_closed is None):
            raise ValueError('close_event and is_closed must be supplied together')
        if inspect.iscoroutinefunction(on_effects) or inspect.iscoroutinefunction(on_failure):
            raise TypeError('actor routing and failure mapping must be synchronous')
        actor = _Actor(name, machine, initializer, deepcopy(arguments or {}), on_effects,
                       on_failure, close_event, is_closed)
        actor.ref = ActorRef(self, actor)
        self._actors[name] = actor
        actor.task = self.loop.create_task(self._run(actor), name='invariant:' + name)
        return actor.ref

    def _submit(self, actor, request):
        def enqueue():
            if actor.fault is not None:
                _answer(request.reply, error=ActorFailed(actor.fault))
            elif self._closed or actor.stopping or actor.stopped.done():
                _answer(request.reply, error=ClosedError('actor is closed'))
            else:
                actor.inbox.put_nowait(request)
                self._changed.set()
        try:
            if threading.get_ident() == self._thread:
                enqueue()
            else:
                self.loop.call_soon_threadsafe(enqueue)
        except RuntimeError as error:
            _answer(request.reply, error=ClosedError(str(error)))
        return request.reply

    def _require_ref(self, ref):
        self._check_loop()
        if not isinstance(ref, ActorRef) or ref._system is not self:
            raise ValueError('actor belongs to another system')
        if self._closed or ref.closed or ref._actor.stopping or ref._actor.fault is not None:
            raise ClosedError('actor is closed or fenced')
        return ref._actor

    def schedule(self, ref, event, delay):
        actor = self._require_ref(ref)
        validate(event, actor.machine.event_type, 'event')
        if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
            raise ValueError('timer delay must be finite and nonnegative')
        def deliver():
            self._timers.pop(handle, None)
            self._submit(actor, _Request('event', event))
        handle = self.loop.call_later(delay, deliver)
        self._timers[handle] = actor
        return handle

    def run_effect(self, ref, awaitable, *, success, failure):
        """Track external work and route its explicit result/failure event to its actor."""
        try:
            actor = self._require_ref(ref)
            if not inspect.isawaitable(awaitable) or not callable(success) or not callable(failure):
                raise TypeError('effect requires an awaitable and explicit success/failure event factories')
        except BaseException:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise
        async def execute():
            recovery = False
            try:
                try:
                    value = await awaitable
                    event = success(value)
                except BaseException as error:
                    recovery = True
                    event = failure(error)
                validate(event, actor.machine.event_type, 'effect receipt')
                receipt = self._submit(actor, _Request('event', event, Future(), recovery))
                await asyncio.wrap_future(receipt)
            except BaseException as error:
                if actor.fault is None:
                    self._fence(actor, 'effect outcome', error)
            finally:
                actor.effects.discard(asyncio.current_task())
                self._changed.set()
        task = self.loop.create_task(execute(), name='invariant-effect:' + actor.name)
        actor.effects.add(task)
        return task

    def _fence(self, actor, stage, error):
        if actor.fault is not None:
            return
        fault = actor.fault = ActorFault(actor.ref, stage, error)
        self.faults.append(fault)
        _answer(actor.initialized, error=ActorFailed(fault))
        if self.on_fault is not None:
            try:
                returned = self.on_fault(fault)
                if inspect.iscoroutine(returned):
                    returned.close()
                    raise TypeError('actor supervisor callback must be synchronous')
            except BaseException as callback_error:
                self.loop.call_exception_handler({'message': 'actor supervisor callback failed',
                                                 'exception': callback_error})
        actor.inbox.put_nowait(_Request('stop'))
        self._changed.set()

    def _route(self, actor, event, effects, state, recovery):
        if actor.on_effects is None:
            return
        try:
            returned = actor.on_effects(actor.ref, effects, state)
            if inspect.iscoroutine(returned):
                returned.close()
                raise TypeError('effect routing must not await physical work')
        except BaseException as error:
            if recovery or actor.on_failure is None:
                raise
            failure = actor.on_failure(actor.ref, event, effects, error)
            validate(failure, actor.machine.event_type, 'effect admission failure')
            actor.inbox.put_nowait(_Request('event', failure, recovery=True))
            raise _RoutingFailure(error) from error

    async def _run(self, actor):
        try:
            store = Store(actor.machine, initializer=actor.initializer, arguments=actor.arguments,
                          owner_task=asyncio.current_task())
            _answer(actor.initialized)
            while True:
                request = await actor.inbox.get()
                try:
                    if request.kind == 'stop':
                        break
                    if actor.fault is not None:
                        raise ActorFailed(actor.fault)
                    if request.kind == 'snapshot':
                        _answer(request.reply, store.state)
                    elif request.kind == 'preview':
                        _answer(request.reply, store.preview(request.value))
                    elif request.kind == 'query':
                        name, arguments = request.value
                        _answer(request.reply, store.query(name, **arguments))
                    else:
                        event = request.value
                        if request.kind == 'close':
                            actor.closing = True
                            if actor.close_event is None:
                                actor.model_closed = True
                                continue
                            event = actor.close_event(store.state) if callable(actor.close_event) else actor.close_event
                        effects = store.dispatch(event)
                        self._route(actor, event, effects, store.state, request.recovery)
                        _answer(request.reply, effects)
                    if actor.closing and actor.is_closed is not None:
                        actor.model_closed = bool(actor.is_closed(store.state))
                except _RoutingFailure as error:
                    _answer(request.reply, error=error.error)
                except AdmissionError as error:
                    # Invalid input and failed preconditions have not executed
                    # the reducer. They do not poison a healthy actor.
                    _answer(request.reply, error=error)
                except BaseException as error:
                    _answer(request.reply, error=error)
                    self._fence(actor, 'dispatch/effect routing', error)
                finally:
                    actor.inbox.task_done()
                    self._changed.set()
                # Queue.get may complete immediately; give peer actors and receipts a turn.
                await asyncio.sleep(0)
        except AdmissionError as error:
            actor.model_closed = actor.stopping = True
            _answer(actor.initialized, error=error)
        except BaseException as error:
            self._fence(actor, 'initialization/actor task', error)
        finally:
            while not actor.inbox.empty():
                request = actor.inbox.get_nowait()
                _answer(request.reply, error=ActorFailed(actor.fault) if actor.fault else ClosedError('actor stopped'))
                actor.inbox.task_done()
            _answer(actor.stopped)
            self._changed.set()

    async def close(self):
        """Drain declared model shutdown and registered effects; never cancel physical work."""
        self._check_loop()
        if self._close_task is None:
            self._closing = True
            self._close_task = self.loop.create_task(self._close(), name='invariant-shutdown')
        await asyncio.shield(self._close_task)

    def _begin_close(self, actor):
        if not actor.close_sent:
            actor.close_sent = True
            self._submit(actor, _Request('close'))

    def _request_close(self, actor):
        reply = Future()
        def begin():
            if self._closed:
                _answer(reply)
                return
            if actor.close_task is None:
                actor.close_task = self.loop.create_task(self._close_actor(actor),
                                                         name='invariant-retire:' + actor.name)
                self._retirements.add(actor.close_task)
                actor.close_task.add_done_callback(self._retirements.discard)
            def completed(task):
                try:
                    _answer(reply, task.result())
                except BaseException as error:
                    _answer(reply, error=error)
            actor.close_task.add_done_callback(completed)
        try:
            if self._closed:
                _answer(reply)
            elif threading.get_ident() == self._thread:
                begin()
            else:
                self.loop.call_soon_threadsafe(begin)
        except RuntimeError as error:
            _answer(reply, error=ClosedError(str(error)))
        return reply

    def _cancel_timers(self, actor=None):
        for handle, owner in tuple(self._timers.items()):
            if actor is None or owner is actor:
                handle.cancel()
                del self._timers[handle]

    @staticmethod
    def _drained(actor):
        return ((actor.fault is not None or actor.model_closed) and not actor.effects
                and actor.inbox.empty())

    async def _close_actor(self, actor):
        self._begin_close(actor)
        while True:
            self._changed.clear()
            if self._drained(actor):
                break
            await self._changed.wait()
        self._cancel_timers(actor)
        if not actor.stopped.done():
            actor.stopping = True
            actor.inbox.put_nowait(_Request('stop'))
        await actor.task
        if self._actors.get(actor.name) is actor:
            del self._actors[actor.name]
        self._changed.set()

    async def _close(self):
        for actor in self._actors.values():
            self._begin_close(actor)
        while True:
            self._changed.clear()
            if all(self._drained(actor) for actor in self._actors.values()):
                break
            await self._changed.wait()
        self._cancel_timers()
        for actor in self._actors.values():
            if not actor.stopped.done():
                actor.stopping = True
                actor.inbox.put_nowait(_Request('stop'))
        await asyncio.gather(*(actor.task for actor in self._actors.values()))
        # Per-reference close callers need their drain operation to complete
        # before a host stops this scheduler immediately after system.close().
        await asyncio.gather(*tuple(self._retirements))
        self._actors.clear()
        self._closed = True


class _RoutingFailure(Exception):
    def __init__(self, error):
        self.error = error
