"""One-shot authority and physical retirement for one frozen scope input.

The adapter owns exact native identities and primary/reference batch adoption.
Each input has its own actor. A candidate publication does not release its
snapshot lease; only an independently validated retirement receipt does that.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    request: int = 0
    source: int = 0
    target: int = 0
    phase: Literal['idle', 'automatic', 'authorized', 'capturing', 'retiring', 'ready', 'stale', 'failed', 'closed'] = 'idle'
    authority: bool = False
    physical: bool = False
    payload: int = 0
    outcome: Literal['ready', 'stale', 'failed'] = 'stale'
    closing: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['automatic', 'request', 'start', 'captured', 'failed', 'invalidate', 'cancel', 'retired', 'close']
    generation: int = 1
    request: int = 0
    source: int = 0
    target: int = 0
    payload: int = 0
    validated: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'sample', 'prepare', 'capture', 'publish', 'cancel', 'release'] = 'none'
    request: int = 0
    source: int = 0
    target: int = 0
    payload: int = 0


def new(generation: int = 1) -> State:
    return State(generation=generation)


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed' or event.generation != state.generation:
        return state, Effects()
    if event.kind == 'close':
        return replace(state, closing=True, authority=False,
                       phase='retiring' if state.physical else 'closed'), Effects(
            True, 'cancel' if state.physical else 'none', state.request, state.source, state.target)
    if state.closing:
        if (event.kind == 'retired' and event.validated and state.physical
                and event.request == state.request and event.source == state.source
                and event.target == state.target):
            return replace(state, physical=False, phase='closed'), Effects(
                True, 'release', state.request, state.source, state.target)
        return state, Effects()
    if event.kind in ('automatic', 'request'):
        if (state.phase in ('idle', 'ready', 'stale', 'failed')
                and event.request > state.request and event.source > 0 and event.target > 0):
            native = event.kind == 'request'
            return replace(state, request=event.request, source=event.source,
                           target=event.target, payload=0, outcome='stale',
                           authority=native, phase='authorized' if native else 'automatic'), Effects(
                True, 'prepare' if native else 'sample', event.request, event.source, event.target)
        return state, Effects()
    if (event.request != state.request or event.source != state.source
            or event.target != state.target):
        return state, Effects()
    if event.kind == 'start' and state.phase == 'authorized' and state.authority:
        return replace(state, authority=False, physical=True, phase='capturing'), Effects(
            True, 'capture', state.request, state.source, state.target)
    if (event.kind == 'captured' and state.phase in ('automatic', 'capturing')
            and event.validated and event.payload > 0):
        return replace(state, payload=event.payload, outcome='ready',
                       phase='retiring' if state.physical else 'ready'), Effects(
            True, 'publish', state.request, state.source, state.target, event.payload)
    if event.kind == 'retired' and event.validated and state.physical:
        return replace(state, physical=False, phase=state.outcome), Effects(
            True, 'release', state.request, state.source, state.target)
    if event.kind in ('invalidate', 'cancel') and state.phase != 'idle':
        return replace(state, authority=False, outcome='stale',
                       phase='retiring' if state.physical else 'stale'), Effects(
            True, 'cancel' if state.physical else 'none', state.request, state.source, state.target)
    if event.kind == 'failed' and state.phase in ('automatic', 'authorized', 'capturing', 'retiring'):
        return replace(state, authority=False, outcome='failed',
                       phase='retiring' if state.physical else 'failed'), Effects(
            True, 'cancel' if state.physical else 'none', state.request, state.source, state.target)
    return state, Effects()
