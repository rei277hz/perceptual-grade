"""Publication authorization with retained backup and restoration obligations.

Resource/target identities refer to adapter-owned immutable transaction records.
A repair receipt certifies restoration of the baseline, never the failed candidate.
Only an acknowledged native replacement may supersede the old repair obligation.
The compatibility event name ``verified`` acknowledges ordered native operation
success for the exact current producer; it does not require an output-content
scan. ``certified`` records completed publication history, not ongoing validity.
Runtime reconciliation may instead relinquish native ownership without repair.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    operation: int = 0
    generation: int = 1
    resource: int = 0
    target: int = 0
    phase: Literal['idle', 'reserving', 'writing', 'verifying', 'publishing', 'ready', 'repair_needed', 'repairing', 'closed'] = 'idle'
    readable: bool = False
    certified: bool = False
    debt: bool = False
    stopping: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin', 'reserved', 'written', 'verified', 'published', 'failed', 'repair', 'repaired', 'aborted', 'superseded', 'relinquished', 'close']
    operation: int = 0
    generation: int = 1
    resource: int = 0
    target: int = 0
    accepted: bool = True
    readable: bool = False
    validated: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'reserve', 'write', 'verify', 'repair', 'publish', 'quarantine', 'release'] = 'none'
    operation: int = 0
    target: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed':
        return state, Effects()
    # This receipt comes only from the actual pre/post native replacement
    # protocol. A timer reset or logical cancellation cannot supply it.
    if event.kind == 'superseded' and event.generation > state.generation and event.operation == state.operation and event.resource == state.resource and event.target == state.target and event.validated:
        return State(operation=state.operation, generation=event.generation,
                     phase='closed' if state.stopping else 'idle', stopping=state.stopping), Effects(True, 'release', state.operation, state.target)
    if event.kind == 'close':
        if state.debt:
            return replace(state, stopping=True, phase='repair_needed', readable=False,
                           certified=False), Effects(True, 'quarantine', state.operation, state.target)
        return replace(state, stopping=True, phase='closed', certified=False), Effects(accepted=True)
    if event.kind == 'begin' and not state.stopping:
        if state.phase in ('idle', 'ready') and event.operation > state.operation and event.generation == state.generation and event.resource > 0 and event.target > 0:
            return State(event.operation, event.generation, event.resource, event.target,
                         'reserving', False, False, True, False), Effects(True, 'reserve', event.operation, event.target)
        return state, Effects()
    current = event.operation == state.operation and event.generation == state.generation and event.resource == state.resource and event.target == state.target
    if not current:
        return state, Effects()
    # Runtime reconciliation relinquishes ownership of native user-authoritative
    # data. It is neither successful publication nor restoration of a baseline.
    if event.kind == 'relinquished' and state.debt and event.validated:
        return replace(state, phase='closed' if state.stopping else 'idle', readable=False,
                       certified=False, debt=False), Effects(True, 'release', state.operation, state.target)
    if event.kind == 'failed' and state.debt:
        return replace(state, phase='repair_needed', readable=False, certified=False), Effects(True, 'quarantine', state.operation, state.target)
    if event.kind == 'repair' and state.phase == 'repair_needed' and state.target > 0:
        return replace(state, phase='repairing', readable=False, certified=False), Effects(True, 'repair', state.operation, state.target)
    if event.kind == 'repaired' and state.phase == 'repairing' and event.readable and event.validated:
        return replace(state, phase='closed' if state.stopping else 'idle', readable=True,
                       certified=False, debt=False), Effects(True, 'release', state.operation, state.target)
    # For this receipt validation means the adapter verified that no native
    # target mutation began, including failed admission after candidate readback.
    if event.kind == 'aborted' and state.debt and event.validated:
        return replace(state, phase='closed' if state.stopping else 'idle', readable=False,
                       certified=False, debt=False), Effects(True, 'release', state.operation, state.target)
    if state.stopping:
        return state, Effects()
    if event.kind == 'reserved' and state.phase == 'reserving':
        if event.accepted:
            return replace(state, phase='writing'), Effects(True, 'write', state.operation, state.target)
        return replace(state, phase='repair_needed', debt=True), Effects(True, 'quarantine', state.operation, state.target)
    if event.kind == 'written' and state.phase == 'writing':
        return replace(state, phase='verifying'), Effects(True, 'verify', state.operation, state.target)
    if event.kind == 'verified' and state.phase == 'verifying':
        if event.validated and event.readable:
            return replace(state, phase='publishing', readable=True, certified=False, debt=True), Effects(True, 'publish', state.operation, state.target)
        return replace(state, phase='repair_needed', readable=False, certified=False, debt=True), Effects(True, 'quarantine', state.operation, state.target)
    if event.kind == 'published' and state.phase == 'publishing':
        if event.validated and state.readable:
            return replace(state, phase='ready', certified=True, debt=False), Effects(accepted=True)
        return replace(state, phase='repair_needed', certified=False, debt=True), Effects(True, 'quarantine', state.operation, state.target)
    return state, Effects()
