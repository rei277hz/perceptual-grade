"""GPU health probes fenced by attempt identity and worker generation."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accepted: bool


@dataclass(frozen=True)
class Event:
    kind: Literal['probe', 'validated', 'failed']
    attempt: int
    generation: int
    validated: bool


@dataclass(frozen=True)
class State:
    phase: int
    attempt: int
    generation: int


def gpu_available(state: State) -> bool:
    return state.phase == 0


def new() -> State:
    return State(attempt=0, generation=0, phase=0)


def state_valid(state: State) -> bool:
    return True


def event_valid(state: State, event: Event) -> bool:
    return True


def apply_failed(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.attempt > 0
        and event.generation >= state.generation
        and (
            (
                state.phase == 0
                or (
                    (
                        state.phase == 2
                        and event.attempt == state.attempt
                        and event.generation == state.generation
                    )
                )
            )
        )
    )
    return (
        replace(
            state,
            attempt=event.attempt if accepted else state.attempt,
            generation=event.generation if accepted else state.generation,
            phase=1 if accepted else state.phase,
        ),
        Effects(accepted=accepted),
    )


def apply_probe(state: State, event: Event) -> tuple[State, Effects]:
    accepted = state.phase != 2 and event.attempt > 0 and (event.generation > state.generation)
    return (
        replace(
            state,
            attempt=event.attempt if accepted else state.attempt,
            generation=event.generation if accepted else state.generation,
            phase=2 if accepted else state.phase,
        ),
        Effects(accepted=accepted),
    )


def apply_validated(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.phase == 2
        and event.attempt == state.attempt
        and event.generation == state.generation
        and event.validated
    )
    return (
        replace(
            state,
            attempt=event.attempt if accepted else state.attempt,
            generation=event.generation if accepted else state.generation,
            phase=0 if accepted else state.phase,
        ),
        Effects(accepted=accepted),
    )


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'failed':
        return apply_failed(state, event)
    if event.kind == 'probe':
        return apply_probe(state, event)
    if event.kind == 'validated':
        return apply_validated(state, event)
    raise ValueError('Unknown machine event')
