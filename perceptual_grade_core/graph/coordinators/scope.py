from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    request: int = 0
    source: int = 0
    phase: Literal['idle', 'capturing', 'registering', 'running', 'retiring', 'ready', 'failed', 'closed'] = 'idle'
    visual: int = 0
    statistics: int = 0
    closing: bool = False
    seed: int = 0
    registered: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['request', 'captured', 'ready', 'needs_seed', 'visual', 'statistics', 'complete', 'retired', 'failed', 'cancel', 'close']
    generation: int = 1
    request: int = 0
    source: int = 0
    payload: int = 0
    validated: bool = False
    register: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'capture', 'register', 'compute', 'publish_visual', 'publish_statistics', 'retire', 'cancel'] = 'none'
    request: int = 0
    payload: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed' or event.generation != state.generation:
        return state, Effects()
    if event.kind == 'close':
        active = state.phase == 'running' or state.phase == 'registering' or state.phase == 'capturing' or state.phase == 'retiring'
        return replace(state, closing=True, phase='retiring' if active else 'closed'), Effects(True, 'cancel' if active else 'none', state.request)
    if event.kind == 'request' and not state.closing and event.request > state.request and event.source > 0:
        if state.phase == 'idle' or state.phase == 'ready' or state.phase == 'failed':
            return replace(state, request=event.request, source=event.source, phase='capturing', seed=0, registered=False), Effects(True, 'capture', event.request)
        return state, Effects()
    if event.request != state.request or event.source != state.source:
        return state, Effects()
    if event.kind == 'retired' and state.phase == 'retiring' and event.validated:
        return replace(state, phase='closed' if state.closing else 'ready'), Effects(accepted=True)
    if state.closing:
        return state, Effects()
    if event.kind == 'captured' and state.phase == 'capturing' and event.validated and event.payload > 0:
        return replace(state, phase='registering' if event.register else 'running', seed=event.payload), Effects(True, 'register' if event.register else 'compute', state.request, event.payload)
    if event.kind == 'ready' and state.phase == 'registering' and event.validated and state.seed > 0:
        return replace(state, phase='running', registered=True), Effects(True, 'compute', state.request, state.seed)
    if event.kind == 'needs_seed' and state.phase == 'running' and event.validated and state.seed > 0:
        return replace(state, phase='registering', registered=False), Effects(True, 'register', state.request, state.seed)
    if event.kind == 'visual' and state.phase == 'running' and event.validated and event.payload > 0:
        return replace(state, visual=event.payload), Effects(True, 'publish_visual', state.request, event.payload)
    if event.kind == 'statistics' and state.phase == 'running' and event.validated and event.payload > 0:
        return replace(state, statistics=event.payload), Effects(True, 'publish_statistics', state.request, event.payload)
    if event.kind == 'complete' and state.phase == 'running':
        return replace(state, phase='retiring'), Effects(True, 'retire', state.request)
    # A capture is physical helper work too. Even admission failure needs an
    # explicit validated no-worker/retired receipt before its snapshot is freed.
    if (event.kind == 'failed' or event.kind == 'cancel') and state.phase in ('capturing', 'running', 'registering'):
        return replace(state, phase='retiring'), Effects(True, 'cancel', state.request)
    return state, Effects()
