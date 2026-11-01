"""Physical native storage owned by one explicit App, never a module registry."""
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
import weakref
import threading

from ..app import ClosedError, ForeignOwnerError
from ..machine import Machine

ACTIVE, FENCED, REBIND, FAILED = range(4)


def _no_keys(value):
    return None, None, None


def _ready():
    return None


@dataclass(frozen=True, slots=True)
class OwnershipConfig:
    machine: Machine
    keys: object = _no_keys
    require_clean: object = _ready

    def __post_init__(self):
        if not isinstance(self.machine, Machine) or not callable(self.keys) or not callable(self.require_clean):
            raise TypeError('Ownership requires a typed machine and explicit native callbacks')


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NativeRef:
    arena: int
    generation: int
    cell: int
    owner: str
    kind: str
    domain: str


class OwnershipObservation:
    """One synchronous pass of actor observations, containing no native values.

    The revision is only a revocation signal. Every cached State came from its
    actor; a boundary never refreshes an old observation into a new generation.
    """
    def __init__(self, context):
        self._context = weakref.ref(context)
        self._revision = context._observation_revision
        self._thread = threading.get_ident()
        self._states = {}
        self._closed = self._entered = False

    def _current_context(self):
        context = self._context()
        if self._closed or context is None:
            raise ReferenceError('Native ownership observation is closed')
        if threading.get_ident() != self._thread:
            raise RuntimeError('Native ownership observation belongs to another thread')
        context.app.require_open()
        if context._observation_revision != self._revision:
            raise ReferenceError('Native ownership observation was revoked')
        return context

    def check(self):
        context = self._current_context()
        if not self._entered:
            raise RuntimeError('Native ownership observation has not been entered')
        return context

    def _state(self, arena):
        context = self.check()
        if arena.context is not context or context._arenas.get(arena.identity) is not arena:
            raise ForeignOwnerError('Observed arena belongs to another ownership context')
        if arena.identity not in self._states:
            state = arena.store.state
            self.check()
            self._states[arena.identity] = state
        return self._states[arena.identity]

    def resolve(self, ref):
        context = self.check()
        return context._resolve(ref, self)

    def generation_key(self):
        context = self.check()
        arena = context.default
        return arena.identity, self._state(arena).epoch

    def __enter__(self):
        context = self._current_context()
        if self._entered:
            raise RuntimeError('Native ownership observation cannot be entered twice')
        self._entered = True
        context._observations.append(self)
        return self

    def __exit__(self, *_error):
        context = self._context()
        if context is not None and self in context._observations:
            context._observations.remove(self)
        self._closed = True
        self._states.clear()


def _retire_owned_cell(controller, generation, cell):
    import threading
    arena = controller()
    if arena is None or cell not in arena._cells:
        return
    if threading.current_thread() is not threading.main_thread():
        arena.context.app._defer_finalizer(_retire_owned_cell, controller, generation, cell)
        return
    arena.drop(generation, cell)


class Arena:
    def __init__(self, context):
        if context is None or getattr(getattr(context, 'app', None), 'ownership', None) is not context:
            raise TypeError('Arena requires its original ownership context')
        context.app.require_open()
        if context._boundaries or any(not arena.store.query('admits', action='discover')
                                      for arena in context.snapshots()):
            raise RuntimeError('Cannot create an RNA arena during a pending ownership boundary')
        self.context = context
        self.identity = next(context._ids)
        self.store = context.app.store(machine=context.config.machine)
        self._cells = {}
        self._native_keys = {}
        self._next_cell = count(1)
        self._borrows = {}
        self._next_borrow = count(1)
        self._closed = False
        context._arenas[self.identity] = self

    @property
    def state(self):
        return self.store.state

    def _event(self, kind, token=0):
        return self.store.machine.event_type(kind=kind, token=token)

    def event(self, kind, token=0):
        cleanup = kind in ('drop', 'release', 'detach', 'fail')
        if self._closed and not cleanup:
            raise ClosedError('RNA arena has been closed')
        if kind in ('begin', 'detach', 'replace', 'defer', 'fail', 'recovered', 'retry'):
            self.context._revoke_observations()
        event = self._event(kind, token)
        effects = self.store._dispatch_cleanup(event) if cleanup else self.store.dispatch(event)
        if not effects.accepted:
            raise RuntimeError('RNA ownership event rejected: ' + kind)
        return self.state

    def retain(self, value, owner):
        self.context.app.require_open()
        if self._closed:
            raise ClosedError('RNA arena has been closed')
        if value is None:
            return None
        if type(owner) is not str:
            raise TypeError('RNA owner must be a value string')
        if not self.store.preview(self._event('retain')).effects.accepted:
            raise RuntimeError('RNA ownership event rejected: retain')
        native_keys = self.context.config.keys(value)
        cell = next(self._next_cell)
        ref = self.context._ref_type(self.identity, self.context._resolution_state(self).epoch, cell, owner, type(value).__name__, self.context.domain)
        # Native identity projection can re-enter the host: dispatch into the
        # current Store and never install the preflight transition snapshot.
        self.event('retain')
        self._cells[cell] = value
        self._native_keys[cell] = native_keys
        weakref.finalize(ref, _retire_owned_cell, weakref.ref(self), ref.generation, cell)
        return ref

    def resolve(self, ref, observation=None):
        if ref is None:
            return None
        self.context._check_ref(ref)
        if self._closed or ref.arena != self.identity:
            return None
        state = observation._state(self) if observation is not None else self.context._resolution_state(self)
        if not self.store.machine.call('resolves', state=state, generation=ref.generation):
            return None
        return self._cells.get(ref.cell)

    def drop(self, generation, cell):
        if cell in self._cells and generation == self.context._resolution_state(self).epoch:
            del self._native_keys[cell]
            del self._cells[cell]
            self.event('drop')

    def detach(self):
        # Clearing the last native wrapper may re-enter host destruction code.
        # Revoke observations before releasing any physical storage.
        self.context._revoke_observations()
        self._native_keys.clear()
        self._cells.clear()
        self.event('detach')

    def begin(self):
        self.event('begin')
        self.detach()
        return self.state.token

    @contextmanager
    def borrow(self, ref):
        self.context.app.require_open()
        value = self.resolve(ref)
        if value is None:
            raise ReferenceError('RNA reference has been retired')
        token = next(self._next_borrow)
        self.event('borrow')
        self._borrows[token] = ref.generation
        try:
            yield value
        finally:
            value = None
            del self._borrows[token]
            self.event('release')

    def native_replaced(self):
        return self.event('replace')

    def recovered(self, token):
        return self.event('recovered', token)

    def fail(self):
        self.event('fail')

    def physical_counts(self):
        return len(self._cells), len(self._borrows)

    def close(self):
        if self._closed:
            return
        self.detach()
        self.fail()
        self._closed = True
        if self.context._arenas.get(self.identity) is self:
            del self.context._arenas[self.identity]
        # Keep an arena with live borrows visible to terminal cleanup only.
        if self._borrows:
            self.context._retired.add(self)


class OwnershipContext:
    def __init__(self, app, config):
        self.app, self.config = app, config
        self.domain = app.identity
        self._arenas, self._ids, self._boundaries = {}, count(1), []
        self._default = None
        self._retired = set()
        self._observations = []
        self._observation_revision = 0
        # Preserve exact classes across an in-place module reload just as the
        # context preserves exact arena objects across Blender callbacks.
        self._ref_type, self._arena_type = NativeRef, Arena

    @property
    def default(self):
        self.app.require_open()
        if self._default is None:
            self._default = self.arena()
        return self._default

    def arena(self):
        return self._arena_type(self)

    def _check_ref(self, ref):
        if type(ref) is not self._ref_type or ref.domain != self.domain:
            raise ForeignOwnerError('Native reference belongs to another App')

    def observation(self):
        """Reuse immutable arena observations until this synchronous pass ends."""
        self.app.require_open()
        return OwnershipObservation(self)

    def _revoke_observations(self):
        self._observation_revision += 1

    def _resolution_state(self, arena):
        if self._observations:
            return self._observations[-1]._state(arena)
        return arena.state

    def retain(self, value, owner):
        return self.default.retain(value, owner)

    def resolve(self, ref):
        return self._resolve(ref)

    def _resolve(self, ref, observation=None):
        if ref is None:
            return None
        self._check_ref(ref)
        arena = self._arenas.get(ref.arena)
        return None if arena is None else arena.resolve(ref, observation)

    def release(self, ref):
        if ref is not None:
            self._check_ref(ref)
            arena = self._arenas.get(ref.arena)
            if arena is not None:
                arena.drop(ref.generation, ref.cell)

    def borrow(self, ref):
        self._check_ref(ref)
        if ref.arena not in self._arenas:
            raise ReferenceError('RNA arena has been retired')
        return self._arenas[ref.arena].borrow(ref)

    def generation(self):
        return self._resolution_state(self.default).epoch

    def generation_key(self):
        return self.default.identity, self._resolution_state(self.default).epoch

    def snapshots(self):
        return tuple(self._arenas.values())

    def retire(self, value):
        selected, _, _ = self.config.keys(value)
        if selected is not None:
            for arena in self.snapshots():
                for cell, (object_key, _, parent) in tuple(arena._native_keys.items()):
                    if selected == object_key or selected == parent:
                        arena.drop(arena.state.epoch, cell)

    def retire_datablock(self, value):
        selected, _, _ = self.config.keys(value)
        if selected is not None:
            for arena in self.snapshots():
                for cell, (object_key, root, _) in tuple(arena._native_keys.items()):
                    if selected == object_key or selected == root:
                        arena.drop(arena.state.epoch, cell)

    def detach_all(self):
        for arena in self.snapshots():
            arena.detach()

    def begin_boundary(self):
        self.app.require_open()
        self._revoke_observations()
        # First-use retention must participate in this boundary too. Creating
        # the lazy default after an outer pre-event would reopen ACTIVE state.
        self.default
        tokens = tuple((arena, arena.begin()) for arena in self.snapshots())
        self._boundaries.append(tokens)
        return tokens

    def _check_tokens(self, tokens):
        tokens = tuple(tokens)
        for arena, token in tokens:
            if type(arena) is not self._arena_type or arena.context is not self or type(token) is not int:
                raise ForeignOwnerError('Recovery receipt belongs to another App')
        return tokens

    def native_replaced(self):
        if not self._boundaries:
            self.begin_boundary()
        tokens = tuple((arena, token) for arena, token in self._boundaries.pop()
                       if self._arenas.get(arena.identity) is arena)
        for arena, _ in tokens:
            arena.native_replaced()
        return tuple((arena, arena.state.token) for arena, _ in tokens)

    def defer_pending_replacement(self):
        if not self._boundaries:
            return False
        for arena in self.snapshots():
            arena.event('defer')
        return True

    def recovered(self, tokens):
        tokens = tuple((arena, token) for arena, token in self._check_tokens(tokens)
                       if self._arenas.get(arena.identity) is arena)
        for arena, token in tokens:
            if not arena.store.preview(arena._event('recovered', token)).effects.accepted:
                raise RuntimeError('Stale or unsafe RNA recovery receipt')
        for arena, token in tokens:
            arena.recovered(token)

    def is_current(self, tokens):
        return all(self._arenas.get(arena.identity) is arena and arena.state.token == token
                   for arena, token in self._check_tokens(tokens))

    def retry_recovery(self, tokens):
        """Reopen a failed rebind pass without replacing native ownership.

        Its original generation, cells and obligations survive. This cannot
        issue a native-replacement receipt or erase an unsafe boundary.
        """
        tokens = self._check_tokens(tokens)
        if self._boundaries or not self.is_current(tokens):
            raise RuntimeError('Recovery retry was superseded by native replacement')
        if {arena.identity for arena, _ in tokens} != {arena.identity for arena in self.snapshots()}:
            raise RuntimeError('Recovery retry must bind every current arena')
        for arena, token in tokens:
            if not arena.store.preview(arena._event('retry', token)).effects.accepted:
                raise RuntimeError('Stale or unsafe recovery retry')
        for arena, token in tokens:
            arena.event('retry', token)
        return tokens

    def admits(self, action):
        return all(arena.store.query('admits', action=action) for arena in self.snapshots())

    def require_ready(self):
        self.app.require_open()
        self.config.require_clean()
        if not self.admits('discover'):
            raise RuntimeError('RNA ownership recovery is incomplete')

    def fail(self):
        for arena in self.snapshots():
            arena.fail()

    def close(self):
        self._revoke_observations()
        for arena in self.snapshots():
            arena.close()
        self._boundaries.clear()
        self._retired = {arena for arena in self._retired if arena._borrows}
        if self._retired:
            raise RuntimeError('Ownership cleanup waits for outstanding scoped borrows')


def bind_holder(holder, context):
    """Bind once; old modal objects must never discover a replacement context."""
    if context is None or getattr(getattr(context, 'app', None), 'ownership', None) is not context:
        raise TypeError('Native holder requires an ownership context')
    previous = holder.__dict__.get('_invariant_ownership')
    if previous is not None and previous is not context:
        raise ForeignOwnerError('Native holder is already bound to another App')
    holder._invariant_ownership = context
    return holder


class OwnedField:
    def __init__(self, owner, default=None):
        self.owner, self.default = owner, default

    def __set_name__(self, cls, name):
        self.storage = '_owned_' + name

    def _context(self, instance):
        context = instance.__dict__.get('_invariant_ownership')
        if context is None or getattr(getattr(context, 'app', None), 'ownership', None) is not context:
            raise TypeError('Bind native holder to its original ownership context first')
        return context

    def __get__(self, instance, cls=None):
        if instance is None:
            return self.default
        return self._context(instance).resolve(instance.__dict__.get(self.storage))

    def __set__(self, instance, value):
        context = self._context(instance)
        old = instance.__dict__.get(self.storage)
        ref = None if value is None else context.retain(value, self.owner)
        instance.__dict__[self.storage] = ref
        context.release(old)

    def __delete__(self, instance):
        self._context(instance).release(instance.__dict__.pop(self.storage, None))
