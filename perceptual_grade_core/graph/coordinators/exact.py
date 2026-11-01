from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    operation: int = 0
    generation: int = 1
    frame: int = 0
    phase: Literal['idle', 'preparing', 'invoking', 'native', 'settling', 'complete', 'repair_needed', 'cancelled', 'closed'] = 'idle'
    outcome: Literal['none', 'success', 'failed', 'cancelled'] = 'none'
    closing: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin', 'changed', 'prepared', 'started', 'native_done', 'stopped', 'failed', 'repair', 'next_frame', 'cancel', 'close']
    operation: int = 0
    generation: int = 1
    frame: int = 0
    validated: bool = False
    outcome: Literal['none', 'success', 'failed', 'cancelled'] = 'none'
    next_frame: int = 0


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'prepare', 'invoke', 'observe_stop', 'repair', 'cancel', 'complete'] = 'none'
    operation: int = 0
    frame: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed' or event.generation != state.generation:
        return state, Effects()
    if event.kind == 'close':
        if state.phase == 'repair_needed' and not event.validated:
            return state, Effects()
        active = state.phase == 'invoking' or state.phase == 'native' or state.phase == 'settling'
        return replace(state, closing=True, phase='settling' if active else 'closed', outcome='cancelled'), Effects(True, 'cancel' if active else 'none', state.operation, state.frame)
    if event.kind == 'begin' and not state.closing and event.operation > state.operation and (state.phase == 'idle' or state.phase == 'complete' or state.phase == 'cancelled'):
        return State(event.operation, state.generation, event.frame, 'preparing'), Effects(True, 'prepare', event.operation, event.frame)
    if event.operation != state.operation or event.frame != state.frame:
        return state, Effects()
    if event.kind == 'stopped' and state.phase == 'settling' and event.validated:
        if state.closing:
            return replace(state, phase='closed'), Effects(accepted=True)
        if state.outcome == 'success':
            return replace(state, phase='complete'), Effects(True, 'complete', state.operation, state.frame)
        return replace(state, phase='cancelled' if state.outcome == 'cancelled' else 'repair_needed'), Effects(accepted=True)
    if state.closing:
        return state, Effects()
    if event.kind == 'changed' and (state.phase == 'preparing' or state.phase == 'invoking'):
        changed_frame = event.next_frame
        return replace(state, phase='preparing', frame=changed_frame, outcome='none'), Effects(True, 'prepare', state.operation, changed_frame)
    if event.kind == 'prepared' and state.phase == 'preparing' and event.validated:
        return replace(state, phase='invoking'), Effects(True, 'invoke', state.operation, state.frame)
    if event.kind == 'started' and state.phase == 'invoking':
        return replace(state, phase='native'), Effects(accepted=True)
    if event.kind == 'native_done' and (state.phase == 'native' or state.phase == 'invoking') and event.outcome != 'none':
        return replace(state, phase='settling', outcome=event.outcome), Effects(True, 'observe_stop', state.operation, state.frame)
    if event.kind == 'failed':
        if state.phase == 'settling':
            return replace(state, outcome='failed'), Effects(accepted=True)
        if state.phase == 'native' or state.phase == 'invoking':
            return replace(state, phase='settling', outcome='failed'), Effects(True, 'observe_stop', state.operation, state.frame)
        if state.phase == 'preparing' or state.phase == 'complete' or state.phase == 'cancelled' or state.phase == 'idle' or state.phase == 'repair_needed':
            return replace(state, phase='repair_needed', outcome='failed'), Effects(accepted=True)
    if event.kind == 'repair' and state.phase == 'repair_needed':
        return replace(state, phase='preparing', outcome='none'), Effects(True, 'prepare', state.operation, state.frame)
    if event.kind == 'next_frame' and state.phase == 'complete':
        return replace(state, phase='preparing', frame=event.next_frame, outcome='none'), Effects(True, 'prepare', state.operation, event.next_frame)
    if event.kind == 'cancel':
        if state.phase == 'native' or state.phase == 'invoking':
            return replace(state, phase='settling', outcome='cancelled'), Effects(True, 'cancel', state.operation, state.frame)
        if state.phase == 'preparing':
            return replace(state, phase='cancelled', outcome='cancelled'), Effects(accepted=True)
    return state, Effects()
