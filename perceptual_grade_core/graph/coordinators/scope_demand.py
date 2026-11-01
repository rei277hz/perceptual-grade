"""Latest scope demand, independent of immutable physical capture leases."""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    epoch: int = 1
    live: bool = True
    paused: bool = False
    desired: int = 0
    source: int = 0
    continuity: int = 0
    active: int = 0
    active_source: int = 0
    active_continuity: int = 0
    accepted: int = 0
    failed: int = 0


@dataclass(frozen=True)
class Event:
    kind: Literal['demand', 'start', 'complete', 'failed', 'pause', 'resume', 'invalidate', 'retry', 'close']
    generation: int = 1
    epoch: int = 1
    request: int = 0
    source: int = 0
    continuity: int = 0
    validated: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'capture', 'publish', 'discard', 'cancel'] = 'none'
    request: int = 0
    source: int = 0
    epoch: int = 0


def new(generation: int = 1) -> State:
    return State(generation=generation)


def step(state: State, event: Event) -> tuple[State, Effects]:
    if not state.live or event.generation != state.generation:
        return state, Effects()
    if event.kind in ('pause', 'invalidate', 'close'):
        return replace(state, epoch=state.epoch + 1, live=event.kind != 'close',
                       paused=event.kind == 'pause' or state.paused,
                       desired=0, source=0, continuity=0, active=0,
                       active_source=0, active_continuity=0, accepted=0, failed=0), Effects(
            True, 'cancel', state.active, state.active_source, state.epoch)
    if event.kind == 'resume':
        return replace(state, paused=False), Effects(True)
    if state.paused or event.epoch != state.epoch:
        return state, Effects()
    if event.kind == 'demand':
        if event.request <= state.desired or event.source <= 0 or event.continuity <= 0:
            return state, Effects()
        return replace(state, desired=event.request, source=event.source,
                       continuity=event.continuity), Effects(True)
    if event.kind == 'retry':
        return replace(state, failed=0), Effects(True)
    if event.kind == 'start':
        if state.active != 0 or state.desired <= state.accepted or state.desired == state.failed:
            return state, Effects()
        return replace(state, active=state.desired, active_source=state.source,
                       active_continuity=state.continuity), Effects(
            True, 'capture', state.desired, state.source, state.epoch)
    if (event.request != state.active or event.source != state.active_source or state.active == 0):
        return state, Effects()
    if event.kind == 'complete' and event.validated:
        compatible = state.active_continuity == state.continuity and state.active > state.accepted
        return replace(state, active=0, active_source=0, active_continuity=0,
                       accepted=state.active if compatible else state.accepted), Effects(
            True, 'publish' if compatible else 'discard', event.request, event.source, state.epoch)
    if event.kind == 'failed':
        return replace(state, failed=state.active, active=0, active_source=0,
                       active_continuity=0), Effects(True, 'discard', event.request, event.source, state.epoch)
    return state, Effects()
