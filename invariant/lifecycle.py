from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int
    busy: bool
    leases: int
    terminal: bool
    cancelling: bool
    stopping: bool
    closed: bool


@dataclass(frozen=True)
class Event:
    kind: Literal['start', 'cancel', 'terminal', 'quiescent', 'died', 'close']
    generation: int


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    start: bool = False
    cancel: bool = False
    reclaim: bool = False


def new() -> State:
    return State(1, False, 0, False, False, False, False)


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.generation != state.generation or state.closed:
        return state, Effects()
    if event.kind == 'start':
        accepted = not state.busy and not state.stopping
        if accepted:
            return replace(state, busy=True, leases=1, terminal=False,
                           cancelling=False), Effects(accepted=True, start=True)
        return state, Effects()
    if event.kind == 'cancel':
        if state.busy:
            return replace(state, cancelling=True), Effects(accepted=True, cancel=True)
        return state, Effects()
    if event.kind == 'terminal':
        if state.busy:
            return replace(state, terminal=True), Effects(accepted=True)
        return state, Effects()
    if event.kind == 'quiescent':
        if state.busy and state.terminal:
            return replace(state, busy=False, leases=0, terminal=False,
                           cancelling=False, closed=state.stopping), Effects(
                               accepted=True, reclaim=True)
        return state, Effects()
    if event.kind == 'died':
        return replace(state, generation=state.generation + 1, busy=False, leases=0,
                       terminal=False, cancelling=False, closed=state.stopping), Effects(
                           accepted=True, reclaim=state.busy)
    return replace(state, stopping=True, cancelling=state.busy,
                   closed=not state.busy), Effects(accepted=True, cancel=state.busy)
