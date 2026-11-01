"""Track native cache invalidation; never certify Image content.

Commit identities describe completed managed publications. Native Images remain
user-authoritative. Dirty, unknown and legacy Images settle without inspection.
Adapter receipts bind observation generations to Image UID and commit identity.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    phase: Literal['idle', 'observing', 'invalidating', 'settled_preserved', 'settled_invalidated', 'removed', 'failed'] = 'idle'
    generation: int = 0
    pending: bool = False
    managed: bool = False
    dirty: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin', 'observe', 'invalidated', 'failed']
    generation: int = 0
    changed: bool = False
    managed: bool = False
    dirty: bool = False
    present: bool = True
    matched: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    invalidate: bool = False
    preserve: bool = False
    settled: bool = False


def new() -> State:
    return State()


def settled(state: State) -> bool:
    return state.phase in ('settled_preserved', 'settled_invalidated', 'removed')


def recovery_required(state: State) -> bool:
    return state.phase in ('observing', 'invalidating', 'failed')


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'begin' and event.generation > state.generation:
        return replace(state, phase='observing', generation=event.generation, dirty=False), Effects(accepted=True)
    if event.generation != state.generation or state.generation == 0:
        return state, Effects()
    if event.kind == 'observe' and state.phase in ('observing', 'invalidating', 'failed'):
        if not event.present:
            return replace(state, phase='removed', pending=False, dirty=False), Effects(True, settled=True)
        if event.dirty or not event.managed:
            return replace(state, phase='settled_preserved', pending=False, managed=event.managed,
                           dirty=event.dirty), Effects(True, preserve=True, settled=True)
        if event.changed or state.pending:
            return replace(state, phase='invalidating', pending=True, managed=True, dirty=False), Effects(True, invalidate=True)
        return replace(state, phase='settled_preserved', pending=False, managed=True, dirty=False), Effects(True, preserve=True, settled=True)
    if event.kind == 'invalidated' and state.phase == 'invalidating' and event.matched:
        return replace(state, phase='settled_invalidated', pending=False), Effects(True, settled=True)
    if event.kind == 'failed' and state.phase in ('observing', 'invalidating'):
        return replace(state, phase='failed', pending=True, dirty=False), Effects(accepted=True)
    return state, Effects()
