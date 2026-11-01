"""Attempt lifetime, shared interests, and physical worker occupancy."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accepted: bool


@dataclass(frozen=True)
class Event:
    kind: Literal['start', 'attach', 'detach', 'retire']
    identity: int
    generation: int


@dataclass(frozen=True)
class State:
    identity: int
    generation: int
    phase: int
    interests: int


def can_admit(jobs: int, size: int, max_jobs: int, max_size: int) -> bool:
    return jobs >= 0 and jobs < max_jobs and (size >= 0) and (size <= max_size)


def can_clean(fenced: bool, attempts: int, helpers: int, payloads: int) -> bool:
    return fenced and attempts == 0 and (helpers == 0) and (payloads == 0)


def new(identity: int, generation: int, interests: int) -> State:
    if not (identity > 0 and generation >= 0 and (interests > 0)):
        raise ValueError('Invalid domain for new')
    return State(generation=generation, identity=identity, interests=interests, phase=0)


def occupied(state: State) -> bool:
    return state.phase == 1 or state.phase == 2


def state_valid(state: State) -> bool:
    return True


def event_valid(state: State, event: Event) -> bool:
    return True


def apply_attach(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.identity == state.identity
        and event.generation == state.generation
        and (
            state.phase == 0 or state.phase == 1
        )
        and state.interests > 0
    )
    return (
        replace(state, interests=state.interests + 1 if accepted else state.interests),
        Effects(accepted=accepted),
    )


def apply_detach(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.identity == state.identity
        and event.generation == state.generation
        and state.phase != 3
        and state.interests > 0
    )
    return (
        replace(
            state,
            interests=state.interests - 1 if accepted else state.interests,
            phase=(
                (
                    (3 if state.phase == 0 else 2) if state.interests == 1 else state.phase
                )
                if accepted
                else state.phase
            ),
        ),
        Effects(accepted=accepted),
    )


def apply_retire(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.identity == state.identity
        and event.generation == state.generation
        and state.phase != 3
    )
    return (
        replace(
            state,
            interests=0 if accepted else state.interests,
            phase=3 if accepted else state.phase,
        ),
        Effects(accepted=accepted),
    )


def apply_start(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        event.identity == state.identity
        and event.generation == state.generation
        and state.phase == 0
        and state.interests > 0
    )
    return (replace(state, phase=1 if accepted else state.phase), Effects(accepted=accepted))


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'attach':
        return apply_attach(state, event)
    if event.kind == 'detach':
        return apply_detach(state, event)
    if event.kind == 'retire':
        return apply_retire(state, event)
    if event.kind == 'start':
        return apply_start(state, event)
    raise ValueError('Unknown machine event')
