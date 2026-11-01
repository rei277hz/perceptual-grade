from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    desired: int = 0
    desired_target: int = 0
    active: int = 0
    target: int = 0
    candidate: int = 0
    phase: Literal['idle', 'admission', 'computing', 'retiring', 'staged', 'publishing', 'ready', 'repair_needed', 'closed'] = 'idle'
    closing: bool = False
    mode: Literal['lut', 'manual'] = 'lut'


@dataclass(frozen=True)
class Event:
    kind: Literal['request', 'admitted', 'preview', 'candidate', 'publish', 'retired', 'published', 'failed', 'retry', 'close']
    generation: int = 1
    revision: int = 0
    target: int = 0
    candidate: int = 0
    accepted: bool = False
    mode: Literal['lut', 'manual'] = 'lut'


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'admit', 'compute', 'cancel', 'preview', 'stage', 'publish', 'repair'] = 'none'
    revision: int = 0
    target: int = 0
    candidate: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed' or event.generation != state.generation:
        return state, Effects()
    if event.kind == 'close':
        active = state.phase in ('admission', 'computing', 'retiring', 'staged', 'publishing', 'repair_needed')
        return replace(state, closing=True, phase='retiring' if active else 'closed'), Effects(True, 'cancel' if active else 'none', state.active, state.target)
    if event.kind == 'request' and not state.closing and event.revision > state.desired and event.target > 0:
        # Manual requests own a frozen source/model/selection descriptor. A live
        # edit cannot replace it; a new manual bake requires the old one to end.
        if state.mode == 'manual' and state.phase not in ('idle', 'ready'):
            return state, Effects()
        if event.mode != state.mode and state.phase not in ('idle', 'ready'):
            return state, Effects()
        if state.phase in ('admission', 'computing', 'retiring', 'staged'):
            return replace(state, desired=event.revision, desired_target=event.target, phase='retiring'), Effects(True, 'cancel', state.active, state.target)
        if state.phase == 'publishing':
            return replace(state, desired=event.revision, desired_target=event.target), Effects(accepted=True)
        if state.phase == 'repair_needed':
            return replace(state, desired=event.revision, desired_target=event.target), Effects(True, 'repair', state.active, state.target, state.candidate)
        return replace(state, desired=event.revision, desired_target=event.target, active=event.revision,
                       target=event.target, candidate=0, phase='admission', mode=event.mode), Effects(True, 'admit', event.revision, event.target)
    if event.revision != state.active or event.target != state.target:
        return state, Effects()
    if event.kind == 'retired' and state.phase == 'retiring' and event.accepted:
        if state.closing:
            return replace(state, phase='closed'), Effects(accepted=True)
        return replace(state, active=state.desired, target=state.desired_target, candidate=0, phase='admission'), Effects(True, 'admit', state.desired, state.desired_target)
    if state.closing:
        return state, Effects()
    if event.kind == 'admitted' and state.phase == 'admission':
        if event.accepted:
            return replace(state, phase='computing'), Effects(True, 'compute', state.active, state.target)
        return replace(state, phase='idle'), Effects(accepted=True)
    if event.kind == 'preview' and state.mode == 'lut' and state.phase in ('computing', 'staged', 'publishing', 'ready') and event.candidate > 0 and event.accepted:
        return state, Effects(True, 'preview', state.active, state.target, event.candidate)
    if event.kind == 'candidate' and state.phase == 'computing' and event.candidate > 0 and event.accepted:
        return replace(state, candidate=event.candidate, phase='staged'), Effects(True, 'stage', state.active, state.target, event.candidate)
    if event.kind == 'publish' and state.phase == 'staged' and event.candidate == state.candidate and event.accepted:
        return replace(state, phase='publishing'), Effects(True, 'publish', state.active, state.target, state.candidate)
    if event.kind == 'failed' and state.phase == 'publishing':
        return replace(state, phase='repair_needed'), Effects(True, 'repair', state.active, state.target, state.candidate)
    if event.kind == 'failed' and state.phase == 'computing':
        return replace(state, phase='retiring'), Effects(True, 'cancel', state.active, state.target)
    if event.kind == 'retry' and state.phase == 'repair_needed':
        return state, Effects(True, 'repair', state.active, state.target, state.candidate)
    if event.kind == 'published' and (state.phase == 'publishing' or state.phase == 'repair_needed') and event.accepted and event.candidate == state.candidate:
        if state.desired > state.active:
            return replace(state, active=state.desired, target=state.desired_target, candidate=0, phase='admission'), Effects(True, 'admit', state.desired, state.desired_target)
        return replace(state, phase='ready'), Effects(accepted=True)
    return state, Effects()
