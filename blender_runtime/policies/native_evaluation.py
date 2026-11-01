"""Exact native-reader leases and ordered foreground readiness observations.

Callbacks enqueue observations without waiting on an actor. The adapter proposes
immutable ledger edits; this machine validates and owns their commits. Native
job observations and callback-to-lease matching remain physical adapter premises.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    revision: int = 0
    serial: int = 0
    tokens: tuple[int, ...] = ()
    failed: bool = False
    closed: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin', 'finish', 'replace', 'fail', 'close']
    generation: int = 1
    revision: int = 0
    lease: int = 0
    index: int = -1
    proposed_tokens: tuple[int, ...] = ()
    validated: bool = False


@dataclass(frozen=True)
class Effects:
    consumed: bool = False
    accepted: bool = False


def new(tokens: tuple[int, ...] = ()) -> State:
    return State(tokens=tokens)


def appended(before: tuple[int, ...], after: tuple[int, ...], lease: int) -> bool:
    if len(after) != len(before) + 1:
        return False
    if after[len(before)] != lease:
        return False
    for index in range(len(before)):
        if after[index] != before[index]:
            return False
    return True


def removed(before: tuple[int, ...], after: tuple[int, ...], index: int, lease: int) -> bool:
    if not 0 <= index < len(before):
        return False
    if before[index] != lease or len(after) != len(before) - 1:
        return False
    for offset in range(len(after)):
        if after[offset] != before[offset + (1 if offset >= index else 0)]:
            return False
    return True


def available(state: State, generation: int, revision: int, native_busy: bool = False) -> bool:
    return (generation == state.generation and revision == state.revision
            and len(state.tokens) == 0 and not state.failed and not state.closed
            and not native_busy)


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.revision != state.revision + 1:
        return state, Effects()
    next_state = replace(state, revision=event.revision)
    if event.generation != state.generation or state.closed:
        return next_state, Effects(consumed=True)
    if event.kind == 'close':
        return replace(next_state, closed=True), Effects(True, True)
    if event.kind == 'fail':
        return replace(next_state, failed=True), Effects(True, True)
    if event.kind == 'replace':
        if event.validated and len(event.proposed_tokens) == 0:
            return replace(next_state, generation=state.generation + 1,
                           tokens=event.proposed_tokens, failed=False), Effects(True, True)
        return next_state, Effects(consumed=True)
    if event.kind == 'begin':
        if event.lease > state.serial and appended(state.tokens, event.proposed_tokens, event.lease):
            return replace(next_state, serial=event.lease,
                           tokens=event.proposed_tokens), Effects(True, True)
        return replace(next_state, failed=True), Effects(consumed=True)
    if event.lease not in state.tokens:
        return next_state, Effects(consumed=True)
    if event.validated and removed(state.tokens, event.proposed_tokens, event.index, event.lease):
        return replace(next_state, tokens=event.proposed_tokens), Effects(True, True)
    return replace(next_state, failed=True), Effects(consumed=True)
