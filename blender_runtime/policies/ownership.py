"""Native ownership epochs, admission, and quiescent reclamation."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accepted: bool


@dataclass(frozen=True)
class Event:
    kind: Literal[
        'begin',
        'retain',
        'drop',
        'borrow',
        'release',
        'detach',
        'replace',
        'defer',
        'recovered',
        'retry',
        'fail',
    ]
    token: int


@dataclass(frozen=True)
class State:
    epoch: int
    phase: int
    cells: int
    borrows: int
    token: int
    unsafe: bool


def admits(state: State, action: int | str) -> bool:
    return (
        action == 'timer'
        or (
            not state.unsafe and (state.phase == 0 or (state.phase == 2 and action == 'rebind'))
        )
    )


def resolves(state: State, generation: int) -> bool:
    """An immutable observation may resolve only its current safe epoch."""
    return generation == state.epoch and not state.unsafe and state.phase in (0, 2)


def _domain_event(state: State, event: Event) -> bool:
    return event.token >= 0


def new() -> State:
    return State(borrows=0, cells=0, epoch=1, phase=0, token=0, unsafe=False)


def _domain_state(state: State) -> bool:
    return valid(state)


def valid(state: State) -> bool:
    return (
        state.epoch >= 1
        and state.token >= 0
        and state.epoch == state.token + 1
        and state.phase >= 0
        and state.phase <= 3
        and state.cells >= 0
        and state.borrows >= 0
    )


def state_valid(state: State) -> bool:
    return _domain_state(state)


def event_valid(state: State, event: Event) -> bool:
    return _domain_event(state, event)


def apply_begin(state: State, event: Event) -> tuple[State, Effects]:
    return (
        replace(state, epoch=state.epoch + 1, phase=1, token=state.token + 1),
        Effects(accepted=True),
    )


def apply_borrow(state: State, event: Event) -> tuple[State, Effects]:
    accepted = not state.unsafe and (state.phase == 0 or state.phase == 2)
    return (
        replace(state, borrows=state.borrows + 1 if accepted else state.borrows),
        Effects(accepted=accepted),
    )


def apply_defer(state: State, event: Event) -> tuple[State, Effects]:
    return (replace(state, phase=1), Effects(accepted=True))


def apply_detach(state: State, event: Event) -> tuple[State, Effects]:
    return (replace(state, cells=0), Effects(accepted=True))


def apply_drop(state: State, event: Event) -> tuple[State, Effects]:
    accepted = state.cells > 0
    return (
        replace(state, cells=state.cells - 1 if accepted else state.cells),
        Effects(accepted=accepted),
    )


def apply_fail(state: State, event: Event) -> tuple[State, Effects]:
    return (replace(state, phase=3), Effects(accepted=True))


def apply_recovered(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.token == state.token
        and state.phase == 2
        and not state.unsafe
        and state.borrows == 0
    )
    return (replace(state, phase=0 if accepted else state.phase), Effects(accepted=accepted))


def apply_retry(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (event.token == state.token and state.phase == 3
                and not state.unsafe and state.borrows == 0)
    return replace(state, phase=2 if accepted else state.phase), Effects(accepted=accepted)


def apply_release(state: State, event: Event) -> tuple[State, Effects]:
    accepted = state.borrows > 0
    return (
        replace(state, borrows=state.borrows - 1 if accepted else state.borrows),
        Effects(accepted=accepted),
    )


def apply_replace(state: State, event: Event) -> tuple[State, Effects]:
    return (
        replace(state, phase=2, unsafe=state.unsafe or state.cells > 0 or state.borrows > 0),
        Effects(accepted=True),
    )


def apply_retain(state: State, event: Event) -> tuple[State, Effects]:
    accepted = not state.unsafe and (state.phase == 0 or state.phase == 2)
    return (
        replace(state, cells=state.cells + 1 if accepted else state.cells),
        Effects(accepted=accepted),
    )


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'begin':
        return apply_begin(state, event)
    if event.kind == 'borrow':
        return apply_borrow(state, event)
    if event.kind == 'defer':
        return apply_defer(state, event)
    if event.kind == 'detach':
        return apply_detach(state, event)
    if event.kind == 'drop':
        return apply_drop(state, event)
    if event.kind == 'fail':
        return apply_fail(state, event)
    if event.kind == 'recovered':
        return apply_recovered(state, event)
    if event.kind == 'retry':
        return apply_retry(state, event)
    if event.kind == 'release':
        return apply_release(state, event)
    if event.kind == 'replace':
        return apply_replace(state, event)
    if event.kind == 'retain':
        return apply_retain(state, event)
    raise ValueError('Unknown machine event')
