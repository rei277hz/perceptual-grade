from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    job: int = 0
    generation: int = 0
    attempt: int = 0
    phase: Literal['idle', 'starting', 'running', 'retiring', 'retry_wait', 'complete', 'failed', 'cancelled', 'closed'] = 'idle'
    gpu: bool = False
    terminal: bool = False
    outcome: Literal['none', 'success', 'failed', 'crashed', 'cancelled'] = 'none'
    retries: int = 0
    stopping: bool = False
    retryable: bool = True


@dataclass(frozen=True)
class Event:
    kind: Literal['submit', 'started', 'complete', 'failed', 'crashed', 'retired', 'retry_decision', 'rebind', 'rejected', 'cancel', 'close']
    job: int = 0
    generation: int = 0
    attempt: int = 0
    gpu: bool = False
    accepted: bool = False
    retryable: bool = True


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'start', 'cancel', 'retire', 'ask_retry', 'fallback', 'deliver', 'fail'] = 'none'
    attempt: int = 0


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if state.phase == 'closed':
        return state, Effects()
    if event.kind == 'close':
        active = state.phase == 'starting' or state.phase == 'running' or state.phase == 'retiring'
        return replace(state, stopping=True, phase='retiring' if active else 'closed',
                       outcome='cancelled' if active else state.outcome), Effects(True, 'cancel' if active else 'none', state.attempt)
    if event.kind == 'submit' and not state.stopping:
        if (state.phase == 'idle' or state.phase == 'complete' or state.phase == 'failed' or state.phase == 'cancelled') and event.job > state.job and event.generation > 0 and event.attempt > 0:
            return State(event.job, event.generation, event.attempt, 'starting', event.gpu,
                         retryable=event.retryable), Effects(True, 'start', event.attempt)
        return state, Effects()
    if event.job != state.job or event.generation != state.generation or event.attempt != state.attempt:
        return state, Effects()
    if event.kind == 'rebind' and state.phase == 'starting' and event.accepted:
        # A confirmed pre-launch replacement consumes its separate startup
        # budget, not the numerical job's crash-retry budget.
        return replace(state, generation=state.generation + 1, attempt=state.attempt + 1,
                       terminal=False, outcome='none'), Effects(True, 'start', state.attempt + 1)
    if event.kind == 'rejected' and state.phase == 'starting' and event.accepted:
        # The adapter confirms no worker received this attempt's payload.
        return replace(state, phase='failed', terminal=True, outcome='failed'), Effects(True, 'fail', state.attempt)
    if event.kind == 'started' and state.phase == 'starting':
        return replace(state, phase='running'), Effects(accepted=True)
    if event.kind == 'cancel' and (state.phase == 'starting' or state.phase == 'running'):
        return replace(state, phase='retiring', outcome='cancelled'), Effects(True, 'cancel', state.attempt)
    if (event.kind == 'complete' or event.kind == 'failed' or event.kind == 'crashed') and (state.phase == 'starting' or state.phase == 'running' or state.phase == 'retiring'):
        outcome = 'cancelled' if state.outcome == 'cancelled' else 'success' if event.kind == 'complete' else 'crashed' if event.kind == 'crashed' else 'failed'
        return replace(state, phase='retiring', terminal=True, outcome=outcome), Effects(True, 'retire', state.attempt)
    if event.kind == 'retired' and state.phase == 'retiring' and event.accepted:
        if state.stopping:
            return replace(state, phase='closed'), Effects(accepted=True)
        if state.outcome == 'cancelled':
            return replace(state, phase='cancelled'), Effects(accepted=True)
        if not state.terminal:
            return state, Effects()
        if state.outcome == 'success':
            return replace(state, phase='complete'), Effects(True, 'deliver', state.attempt)
        if state.gpu and state.retryable:
            return replace(state, phase='starting', gpu=False, attempt=state.attempt + 1, generation=state.generation + 1, terminal=False, outcome='none'), Effects(True, 'fallback', state.attempt + 1)
        if state.retryable and state.outcome == 'crashed' and state.retries < 1:
            return replace(state, phase='retry_wait'), Effects(True, 'ask_retry', state.attempt)
        return replace(state, phase='failed'), Effects(True, 'fail', state.attempt)
    if event.kind == 'retry_decision' and state.phase == 'retry_wait':
        if event.accepted and state.retryable and state.retries < 1:
            return replace(state, phase='starting', retries=state.retries + 1, attempt=state.attempt + 1, generation=state.generation + 1, terminal=False, outcome='none'), Effects(True, 'start', state.attempt + 1)
        return replace(state, phase='failed'), Effects(True, 'fail', state.attempt)
    return state, Effects()
