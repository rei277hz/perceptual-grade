"""One crash retry per job; rebinding does not consume its budget."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accepted: bool


@dataclass(frozen=True)
class Event:
    kind: Literal['crashed', 'rebind', 'failed']


@dataclass(frozen=True)
class State:
    retries: int


def new(retries: int) -> State:
    if not (retries >= 0 and retries <= 1):
        raise ValueError('Invalid domain for new')
    return State(retries=retries)


def state_valid(state: State) -> bool:
    return True


def event_valid(state: State, event: Event) -> bool:
    return True


def apply_crashed(state: State, event: Event) -> tuple[State, Effects]:
    accepted = state.retries >= 0 and state.retries < 1
    return (
        replace(state, retries=state.retries + 1 if accepted else state.retries),
        Effects(accepted=accepted),
    )


def apply_failed(state: State, event: Event) -> tuple[State, Effects]:
    return (state, Effects(accepted=False))


def apply_rebind(state: State, event: Event) -> tuple[State, Effects]:
    return (state, Effects(accepted=True))


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'crashed':
        return apply_crashed(state, event)
    if event.kind == 'failed':
        return apply_failed(state, event)
    if event.kind == 'rebind':
        return apply_rebind(state, event)
    raise ValueError('Unknown machine event')
