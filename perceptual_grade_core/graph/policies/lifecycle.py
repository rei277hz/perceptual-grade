"""Session admission through activation, save, recovery, and shutdown."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accept: bool = False
    start: bool = False
    cancel: bool = False
    replace: bool = False
    drop: bool = False
    redraw: bool = False
    effect: bool = False
    failed: bool = False
    recovered: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal[
        'activate',
        'quiesce',
        'save_begin',
        'save_ready',
        'save_done',
        'recover',
        'recovered',
        'fail',
        'crash',
        'disable',
    ]


@dataclass(frozen=True)
class LifecycleState:
    phase: int = 0
    dirty: bool = False
    error: bool = False


def lifecycle_admits(state: LifecycleState, action: int | str) -> bool:
    return (
        state.phase == 0
        or state.phase == 1
        or state.phase == 2
        or action == 'timer'
        or (
            state.phase == 3 and action == 'rebind'
        )
    )


def new() -> LifecycleState:
    return LifecycleState()


def _domain_state(state: LifecycleState) -> bool:
    return (
        state.phase >= 0
        and state.phase <= 5
        and (
            not state.phase == 0 or (not state.dirty and (not state.error))
        )
        and (
            not (state.phase == 1 or state.phase == 3) or (state.dirty and (not state.error))
        )
        and (
            not state.phase == 4 or (state.dirty and state.error)
        )
        and (
            not state.phase == 5 or (not state.dirty and (not state.error))
        )
        and (
            not state.phase == 2 or not state.error
        )
    )


def state_valid(state: LifecycleState) -> bool:
    return _domain_state(state)


def event_valid(state: LifecycleState, event: Event) -> bool:
    return True


def apply_activate(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 0 or state.phase == 5
    return (
        replace(
            state,
            dirty=False if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=0 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_crash(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    return (replace(state, dirty=True, error=False, phase=3), Effects(accept=True, recovered=True))


def apply_disable(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    return (replace(state, dirty=False, error=False, phase=5), Effects(accept=True))


def apply_fail(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    return (replace(state, dirty=True, error=True, phase=4), Effects(accept=True, failed=True))


def apply_quiesce(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 0 or state.phase == 1
    return (
        replace(
            state,
            dirty=True if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=1 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_recover(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    return (replace(state, dirty=True, error=False, phase=3), Effects(accept=True, recovered=True))


def apply_recovered(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 3
    return (
        replace(
            state,
            dirty=False if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=0 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_save_begin(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 0 or state.phase == 1
    return (
        replace(
            state,
            dirty=True if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=2 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_save_done(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 2 and (not state.dirty)
    return (
        replace(
            state,
            dirty=False if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=0 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_save_ready(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    accepted = state.phase == 2
    return (
        replace(
            state,
            dirty=False if accepted else state.dirty,
            error=False if accepted else state.error,
            phase=2 if accepted else state.phase,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def step(state: LifecycleState, event: Event) -> tuple[LifecycleState, Effects]:
    if event.kind == 'activate':
        return apply_activate(state, event)
    if event.kind == 'crash':
        return apply_crash(state, event)
    if event.kind == 'disable':
        return apply_disable(state, event)
    if event.kind == 'fail':
        return apply_fail(state, event)
    if event.kind == 'quiesce':
        return apply_quiesce(state, event)
    if event.kind == 'recover':
        return apply_recover(state, event)
    if event.kind == 'recovered':
        return apply_recovered(state, event)
    if event.kind == 'save_begin':
        return apply_save_begin(state, event)
    if event.kind == 'save_done':
        return apply_save_done(state, event)
    if event.kind == 'save_ready':
        return apply_save_ready(state, event)
    raise ValueError('Unknown machine event')
