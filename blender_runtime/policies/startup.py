"""Per-lane helper startup failure budget and successful dispatch reset."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accepted: bool


@dataclass(frozen=True)
class Event:
    kind: Literal['dispatch', 'failed']


@dataclass(frozen=True)
class State:
    failures: int


def new(failures: int) -> State:
    if not (failures >= 0 and failures < 2):
        raise ValueError('Invalid domain for new')
    return State(failures=failures)


def state_valid(state: State) -> bool:
    return True


def event_valid(state: State, event: Event) -> bool:
    return True


def apply_dispatch(state: State, event: Event) -> tuple[State, Effects]:
    return (replace(state, failures=0), Effects(accepted=True))


def apply_failed(state: State, event: Event) -> tuple[State, Effects]:
    accepted = state.failures + 1 < 2
    return (replace(state, failures=state.failures + 1), Effects(accepted=accepted))


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'dispatch':
        return apply_dispatch(state, event)
    if event.kind == 'failed':
        return apply_failed(state, event)
    raise ValueError('Unknown machine event')
