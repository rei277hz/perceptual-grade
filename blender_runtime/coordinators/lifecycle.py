from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 1
    operation: int = 0
    phase: Literal['active', 'draining', 'replacing', 'inspecting', 'repair_needed', 'repairing', 'releasing', 'closed'] = 'active'
    closing: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin', 'drained', 'replaced', 'external_replaced', 'inspect', 'inspected', 'failed', 'repair', 'repaired', 'released', 'close']
    operation: int = 0
    generation: int = 1
    workers: int = 0
    borrows: int = 0
    validated: bool = False
    readers: int = 0
    debts: int = 0


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'fence', 'replace', 'inspect', 'repair', 'activate', 'release'] = 'none'
    generation: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed':
        return state, Effects()
    if event.kind == 'close':
        return replace(state, closing=True, phase='draining'), Effects(True, 'fence', state.generation)
    if event.kind == 'begin' and not state.closing and event.operation > state.operation and event.generation == state.generation:
        return replace(state, operation=event.operation, phase='draining'), Effects(True, 'fence', state.generation)
    if event.operation != state.operation or event.generation != state.generation:
        return state, Effects()
    if event.kind == 'inspect' and state.phase == 'active' and not state.closing:
        return replace(state, phase='inspecting'), Effects(True, 'inspect', state.generation)
    if event.kind == 'external_replaced' and state.phase in ('draining', 'replacing') and not state.closing and event.validated:
        return replace(state, phase='inspecting', generation=state.generation + 1), Effects(True, 'inspect', state.generation + 1)
    if event.kind == 'drained' and state.phase == 'draining' and event.readers == 0 and event.borrows == 0 and event.debts == 0 and event.validated:
        if state.closing:
            if event.workers == 0:
                return replace(state, phase='releasing'), Effects(True, 'release', state.generation)
            return state, Effects()
        return replace(state, phase='replacing'), Effects(True, 'replace', state.generation)
    if event.kind == 'replaced' and state.phase == 'replacing' and event.validated:
        return replace(state, phase='inspecting', generation=state.generation + 1), Effects(True, 'inspect', state.generation + 1)
    if event.kind == 'failed' and state.phase != 'active':
        return replace(state, phase='repair_needed'), Effects(accepted=True)
    if event.kind == 'repair' and state.phase == 'repair_needed':
        return replace(state, phase='repairing'), Effects(True, 'repair', state.generation)
    if event.kind == 'repaired' and state.phase == 'repairing' and event.validated and event.debts == 0:
        if state.closing:
            return replace(state, phase='draining'), Effects(True, 'fence', state.generation)
        return replace(state, phase='inspecting'), Effects(True, 'inspect', state.generation)
    if event.kind == 'released' and state.phase == 'releasing' and state.closing and event.validated:
        return replace(state, phase='closed'), Effects(accepted=True)
    if event.kind == 'inspected' and state.phase == 'inspecting':
        if event.validated and not state.closing:
            if event.readers == 0 and event.borrows == 0 and event.debts == 0:
                return replace(state, phase='active'), Effects(True, 'activate', state.generation)
            return state, Effects(accepted=True)
        return replace(state, phase='repair_needed'), Effects(accepted=True)
    return state, Effects()
