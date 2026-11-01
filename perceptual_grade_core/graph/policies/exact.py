"""Exact-operation validation, native execution, settlement, and cancellation."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class ExactEffects:
    accept: bool = False
    invoke: bool = False
    cancel: bool = False
    advance: bool = False


@dataclass(frozen=True)
class ExactEvent:
    kind: Literal[
        'begin',
        'changed',
        'validated',
        'native_begin',
        'native_result',
        'native_stopped',
        'cleanup_failed',
        'settled',
        'next_frame',
        'cancel',
        'fail',
    ]
    operation: int = 0
    session: int = 0
    revision: int = 0
    frame: int = 0
    key: int = 0
    token: int = 0
    resolution: int = 0
    outcome: int = 0


@dataclass(frozen=True)
class ExactOperationState:
    phase: int = 0
    operation: int = 0
    session: int = 0
    revision: int = 0
    frame: int = 0
    required: int = 0
    receipt: int = 0
    lease: int = 0
    outcome: int = 0
    cancelled: bool = False
    stopped: bool = False


def _domain_event(state: ExactOperationState, event: ExactEvent) -> bool:
    return exact_domain(event)


def exact_domain(event: ExactEvent) -> bool:
    return (
        event.operation >= 0
        and event.session >= 0
        and event.revision >= 0
        and event.frame >= 0
        and event.key >= 0
        and event.token >= 0
        and event.resolution >= 0
        and event.outcome >= 0
    )


def exact_invariant(state: ExactOperationState) -> bool:
    return (
        state.phase >= 0
        and state.phase <= 7
        and state.operation >= 0
        and state.session >= 0
        and state.revision >= 0
        and state.frame >= 0
        and state.required >= 0
        and state.receipt >= 0
        and state.lease >= 0
        and state.outcome >= 0
        and state.outcome <= 3
        and (
            (
                not state.phase != 0
                or (
                    state.operation > 0 and state.session > 0 and (state.required > 0)
                )
            )
        )
        and (
            (
                not state.phase == 2
                or (
                    state.receipt == state.required and state.receipt > 0 and (not state.cancelled)
                )
            )
        )
        and (
            not state.phase != 2 or state.receipt == 0
        )
        and (state.lease > 0) == (state.phase == 3 or state.phase == 4)
        and (
            not state.phase == 3 or (state.outcome == 0 and (not state.stopped))
        )
        and (
            not state.phase == 4 or state.outcome > 0
        )
        and (
            not state.phase == 5 or (state.outcome == 1 and state.stopped and (not state.cancelled))
        )
    )


def new() -> ExactOperationState:
    return ExactOperationState()


def _domain_state(state: ExactOperationState) -> bool:
    return exact_invariant(state)


def state_valid(state: ExactOperationState) -> bool:
    return _domain_state(state)


def event_valid(state: ExactOperationState, event: ExactEvent) -> bool:
    return _domain_event(state, event)


def apply_begin(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        (
            state.phase == 0 or state.phase >= 5
        )
        and event.operation > 0
        and event.session > 0
        and event.key > 0
    )
    return (
        replace(
            state,
            cancelled=False if accepted else state.cancelled,
            frame=event.frame if accepted else state.frame,
            lease=0 if accepted else state.lease,
            operation=event.operation if accepted else state.operation,
            outcome=0 if accepted else state.outcome,
            phase=1 if accepted else state.phase,
            receipt=0 if accepted else state.receipt,
            required=event.key if accepted else state.required,
            revision=0 if accepted else state.revision,
            session=event.session if accepted else state.session,
            stopped=False if accepted else state.stopped,
        ),
        ExactEffects(accept=accepted),
    )


def apply_cancel(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and (
            state.phase == 1 or state.phase == 2 or state.phase == 3 or (state.phase == 4)
        )
    )
    return (
        replace(
            state,
            cancelled=True if accepted else state.cancelled,
            phase=(
                (
                    7 if state.phase == 1 or state.phase == 2 else state.phase
                )
                if accepted
                else state.phase
            ),
            receipt=0 if accepted else state.receipt,
        ),
        ExactEffects(accept=accepted, cancel=accepted),
    )


def apply_changed(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and (
            state.phase == 1 or state.phase == 2
        )
        and event.key > 0
        and event.key != state.required
    )
    return (
        replace(
            state,
            phase=1 if accepted else state.phase,
            receipt=0 if accepted else state.receipt,
            required=event.key if accepted else state.required,
            revision=state.revision + 1 if accepted else state.revision,
        ),
        ExactEffects(accept=accepted),
    )


def apply_cleanup_failed(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 4
        and event.token == state.lease
        and event.token > 0
    )
    return (replace(state, outcome=2 if accepted else state.outcome), ExactEffects(accept=accepted))


def apply_fail(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and (
            state.phase == 1 or state.phase == 2
        )
    )
    return (
        replace(
            state,
            outcome=2 if accepted else state.outcome,
            phase=6 if accepted else state.phase,
            receipt=0 if accepted else state.receipt,
        ),
        ExactEffects(accept=accepted),
    )


def apply_native_begin(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 2
        and state.receipt == state.required
        and event.key == state.required
        and state.receipt > 0
        and state.lease == 0
        and event.token > 0
        and not state.cancelled
    )
    return (
        replace(
            state,
            lease=event.token if accepted else state.lease,
            outcome=0 if accepted else state.outcome,
            phase=3 if accepted else state.phase,
            receipt=0 if accepted else state.receipt,
            stopped=False if accepted else state.stopped,
        ),
        ExactEffects(accept=accepted, invoke=accepted),
    )


def apply_native_result(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 3
        and event.token == state.lease
        and event.token > 0
        and state.outcome == 0
        and (
            event.outcome == 1 or event.outcome == 2 or event.outcome == 3
        )
    )
    return (
        replace(
            state,
            outcome=event.outcome if accepted else state.outcome,
            phase=4 if accepted else state.phase,
        ),
        ExactEffects(accept=accepted),
    )


def apply_native_stopped(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 4
        and event.token == state.lease
        and event.token > 0
    )
    return (
        replace(state, stopped=True if accepted else state.stopped),
        ExactEffects(accept=accepted),
    )


def apply_next_frame(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 5
        and not state.cancelled
        and event.key > 0
    )
    return (
        replace(
            state,
            frame=state.frame + 1 if accepted else state.frame,
            outcome=0 if accepted else state.outcome,
            phase=1 if accepted else state.phase,
            receipt=0 if accepted else state.receipt,
            required=event.key if accepted else state.required,
            revision=state.revision + 1 if accepted else state.revision,
            stopped=False if accepted else state.stopped,
        ),
        ExactEffects(accept=accepted, advance=accepted),
    )


def apply_settled(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 4
        and state.stopped
        and event.token == state.lease
        and event.token > 0
    )
    return (
        replace(
            state,
            lease=0 if accepted else state.lease,
            phase=(
                (
                    7 if state.cancelled or state.outcome == 3 else 5 if state.outcome == 1 else 6
                )
                if accepted
                else state.phase
            ),
            receipt=0 if accepted else state.receipt,
        ),
        ExactEffects(accept=accepted),
    )


def apply_validated(
    state: ExactOperationState,
    event: ExactEvent,
) -> tuple[ExactOperationState, ExactEffects]:
    accepted = (
        event.operation == state.operation
        and event.session == state.session
        and event.revision == state.revision
        and event.frame == state.frame
        and state.phase == 1
        and event.key == state.required
        and event.key > 0
        and event.resolution == 65
        and not state.cancelled
    )
    return (
        replace(
            state,
            phase=2 if accepted else state.phase,
            receipt=event.key if accepted else state.receipt,
        ),
        ExactEffects(accept=accepted),
    )


def step(state: ExactOperationState, event: ExactEvent) -> tuple[ExactOperationState, ExactEffects]:
    if event.kind == 'begin':
        return apply_begin(state, event)
    if event.kind == 'cancel':
        return apply_cancel(state, event)
    if event.kind == 'changed':
        return apply_changed(state, event)
    if event.kind == 'cleanup_failed':
        return apply_cleanup_failed(state, event)
    if event.kind == 'fail':
        return apply_fail(state, event)
    if event.kind == 'native_begin':
        return apply_native_begin(state, event)
    if event.kind == 'native_result':
        return apply_native_result(state, event)
    if event.kind == 'native_stopped':
        return apply_native_stopped(state, event)
    if event.kind == 'next_frame':
        return apply_next_frame(state, event)
    if event.kind == 'settled':
        return apply_settled(state, event)
    if event.kind == 'validated':
        return apply_validated(state, event)
    raise ValueError('Unknown machine event')
