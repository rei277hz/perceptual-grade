"""Scheduling channels, replacement decisions, and fair scope selection."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class Effects:
    accept: bool = False
    start: bool = False
    cancel: bool = False
    replace: bool = False
    drop: bool = False
    redraw: bool = False
    effect: bool = False
    failed: bool = False
    recovered: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal[
        'begin',
        'request',
        'release',
        'signal',
        'ready',
        'publish',
        'start',
        'complete',
        'replace',
        'invalidate',
        'shutdown',
        'recover',
        'retry',
        'fail',
        'effect_begin',
        'effect_commit',
        'effect_fail',
    ]
    owner: int = 0
    session: int = -1
    key: int = 0
    signal: int = 0
    quality: int = 0
    token: int = 0
    ready: bool = True
    started: bool = False
    preview: bool = False
    finished: bool = False
    published: bool = False
    cancelled: bool = False
    scope_demand: bool = False
    provenance: int = 0
    resolution: int = 0
    phase: int = 0
    epoch: int = 0
    history: bool = False


@dataclass(frozen=True)
class ScopeCycle:
    keep: bool
    cancel: bool
    publish: bool
    collect: bool
    advance: bool
    retire: bool
    dispatch: bool


@dataclass(frozen=True)
class State:
    owner: int = 0
    session: int = 1
    epoch: int = 0
    live: bool = True
    dragging: bool = False
    requested: int = 0
    signal: int = 0
    ready: bool = True
    quality: int = 0
    publications: int = 0
    worker: int = 0
    worker_key: int = 0
    worker_epoch: int = 0
    scope: bool = False
    linear: bool = False
    history: bool = False
    provenance: int = 0
    display_available: bool = False
    display_key: int = 0
    display_signal: int = 0
    display_provenance: int = 0
    display_resolution: int = 0
    display_epoch: int = 0
    error: bool = False
    error_key: int = 0
    effect: int = 0
    effect_phase: int = 0


def _domain_event(state: State, event: Event) -> bool:
    return event_domain(event)


def event_domain(event: Event) -> bool:
    return (
        event.owner >= 0
        and event.session >= -1
        and event.key >= 0
        and event.signal >= 0
        and event.token >= 0
        and event.provenance >= 0
        and event.resolution >= 0
        and event.phase >= 0
        and event.epoch >= 0
    )


def final_scope_resolution(resolutions: tuple[int, ...]) -> bool:
    return _final_scope_resolution_all_resolution(resolutions)


def initial(owner: int, key: int, session: int, scope: bool, linear: bool) -> State:
    return State(linear=linear, owner=owner, requested=key, scope=scope, session=session)


def invariant(state: State) -> bool:
    return (
        state.owner >= 0
        and state.session >= 0
        and state.epoch >= 0
        and state.requested >= 0
        and state.signal >= 0
        and state.quality >= 0
        and state.quality <= 2
        and state.publications >= 0
        and state.publications <= state.quality
        and state.worker >= 0
        and state.worker_key >= 0
        and state.worker_epoch >= 0
        and state.worker_epoch <= state.epoch
        and (state.worker == 0) == (state.worker_key == 0)
        and (
            not state.worker == 0 or state.worker_epoch == 0
        )
        and state.provenance >= 0
        and state.display_key >= 0
        and state.display_signal >= 0
        and state.display_provenance >= 0
        and state.display_resolution >= 0
        and state.display_epoch >= 0
        and state.display_epoch <= state.epoch
        and (
            (
                not state.display_available
                or (
                    (
                        state.scope
                        and state.display_key > 0
                        and state.display_provenance > 0
                        and state.display_resolution == 65
                    )
                )
            )
        )
        and (
            (
                not not state.display_available
                or (
                    (
                        state.display_key == 0
                        and state.display_signal == 0
                        and state.display_provenance == 0
                        and state.display_resolution == 0
                    )
                )
            )
        )
        and state.error_key >= 0
        and state.error == (state.error_key != 0)
        and state.effect >= 0
        and state.effect_phase >= 0
        and (state.effect == 0) == (state.effect_phase == 0)
        and (
            (
                not not state.live
                or (
                    (
                        not state.dragging
                        and state.requested == 0
                        and state.signal == 0
                        and state.quality == 0
                        and state.publications == 0
                        and state.worker == 0
                        and not state.display_available
                        and not state.error
                        and state.effect == 0
                    )
                )
            )
        )
    )


def linear_scope_request(key: int, source: int) -> tuple[int, int, int]:
    return (key, source, 1)


def replacement_decision(
    requested: int,
    active: int,
    dragging: bool,
    started: bool,
    preview: bool,
    finished: bool,
    published: bool,
    cancelled: bool,
    scope_demand: bool,
) -> tuple[bool, bool]:
    obsolete = active != 0 and (active != requested or cancelled)
    protected_drag = (
        started
        and dragging
        and not cancelled
        and (
            not preview or scope_demand
        )
        and not finished
    )
    replaceable = (
        not started
        or (
            finished and (not dragging or published or active == requested or (not preview))
        )
    )
    return (obsolete and (not protected_drag), obsolete and replaceable)


def same_session(state: State, event: Event) -> bool:
    return event.session == -1 or event.session == state.session


def scope_ack_admissible(
    wanted: bool,
    cache_match: bool,
    backend_match: bool,
    session_match: bool,
    generation_match: bool,
) -> bool:
    return wanted and cache_match and backend_match and session_match and generation_match


def scope_cycle(
    wanted: bool,
    compatible: bool,
    exact: bool,
    dragging: bool,
    retired: bool,
    preview: bool,
    control: bool,
    ack: bool,
    pending: bool,
    history: bool = False,
) -> ScopeCycle:
    keep = scope_job_admissible(wanted, compatible, exact, dragging, history)
    advance = keep and retired and control and ack
    retire = retired and (not advance)
    return ScopeCycle(
        advance=advance,
        cancel=not keep,
        collect=keep and retired,
        dispatch=retire and pending,
        keep=keep,
        publish=keep and preview,
        retire=retire,
    )


def scope_job_admissible(wanted: bool, compatible: bool, exact: bool, dragging: bool, history: bool = False) -> bool:
    return wanted and compatible and (exact or dragging or history)


def scope_next(eligible: tuple[bool, ...], start: int) -> int:
    return _cyclic_first(eligible, start)


def _domain_state(state: State) -> bool:
    return invariant(state)


def state_valid(state: State) -> bool:
    return _domain_state(state)


def event_valid(state: State, event: Event) -> bool:
    return _domain_event(state, event)


def apply_begin(state: State, event: Event) -> tuple[State, Effects]:
    return (
        replace(
            state,
            dragging=(
                (
                    True
                )
                if state.live and event.owner == state.owner and same_session(state, event)
                else state.dragging
            ),
        ),
        Effects(),
    )


def apply_complete(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.token != 0
        and event.token == state.worker
    )
    return (
        replace(
            state,
            worker=0 if accepted else state.worker,
            worker_epoch=0 if accepted else state.worker_epoch,
            worker_key=0 if accepted else state.worker_key,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_effect_begin(state: State, event: Event) -> tuple[State, Effects]:
    update_effect = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.token != 0
        and event.phase > 0
        and state.effect == 0
    )
    return (
        replace(
            state,
            effect=event.token if update_effect else state.effect,
            effect_phase=event.phase if update_effect else state.effect_phase,
        ),
        Effects(drop=not update_effect, effect=update_effect),
    )


def apply_effect_commit(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.token != 0
        and event.token == state.effect
        and event.phase == state.effect_phase
    )
    return (
        replace(
            state,
            effect=0 if accepted else state.effect,
            effect_phase=0 if accepted else state.effect_phase,
        ),
        Effects(accept=accepted, drop=not accepted, effect=accepted),
    )


def apply_effect_fail(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.token != 0
        and event.token == state.effect
        and event.phase == state.effect_phase
    )
    return (
        replace(
            state,
            effect=0 if accepted else state.effect,
            effect_phase=0 if accepted else state.effect_phase,
        ),
        Effects(accept=accepted, drop=not accepted, effect=accepted, failed=accepted),
    )


def apply_fail(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.key != 0
        and event.key == state.requested
        and (
            (
                event.token == 0
                or (
                    (
                        state.live
                        and event.owner == state.owner
                        and same_session(state, event)
                        and event.key != 0
                        and event.key == state.requested
                        and event.token != 0
                        and event.token == state.worker
                    )
                )
            )
        )
    )
    update_worker = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.key != 0
        and event.key == state.requested
        and event.token != 0
        and event.token == state.worker
    )
    return (
        replace(
            state,
            error=True if accepted else state.error,
            error_key=event.key if accepted else state.error_key,
            worker=0 if update_worker else state.worker,
            worker_epoch=0 if update_worker else state.worker_epoch,
            worker_key=0 if update_worker else state.worker_key,
        ),
        Effects(accept=accepted, drop=not accepted, failed=accepted),
    )


def apply_invalidate(state: State, event: Event) -> tuple[State, Effects]:
    update_owner = state.live and event.owner == state.owner and same_session(state, event)
    return (
        replace(
            state,
            display_available=False if update_owner else state.display_available,
            display_epoch=0 if update_owner else state.display_epoch,
            display_key=0 if update_owner else state.display_key,
            display_provenance=0 if update_owner else state.display_provenance,
            display_resolution=0 if update_owner else state.display_resolution,
            display_signal=0 if update_owner else state.display_signal,
            dragging=False if update_owner else state.dragging,
            effect=0 if update_owner else state.effect,
            effect_phase=0 if update_owner else state.effect_phase,
            epoch=state.epoch + 1 if update_owner else state.epoch,
            error=False if update_owner else state.error,
            error_key=0 if update_owner else state.error_key,
            live=True if update_owner else state.live,
            owner=state.owner + 1 if update_owner else state.owner,
            provenance=0 if update_owner else state.provenance,
            publications=0 if update_owner else state.publications,
            quality=0 if update_owner else state.quality,
            ready=True if update_owner else state.ready,
            requested=0 if update_owner else state.requested,
            session=state.session + 1 if update_owner else state.session,
            signal=0 if update_owner else state.signal,
            worker=0 if update_owner else state.worker,
            worker_epoch=0 if update_owner else state.worker_epoch,
            worker_key=0 if update_owner else state.worker_key,
        ),
        Effects(cancel=update_owner and state.worker != 0),
    )


def apply_publish(state: State, event: Event) -> tuple[State, Effects]:
    current_request = (event.key != 0 and event.key == state.requested
                       and (not state.scope or event.epoch == state.epoch))
    valid_quality = event.quality == 1 or event.quality == 2
    scope_compatible = (
        not state.scope
        or (
            (
                event.resolution == 65
                and event.provenance != 0
                and event.epoch >= state.display_epoch
                and event.epoch <= state.epoch
                and (
                    not current_request or event.provenance == state.provenance
                )
            )
        )
    )
    settled_progress = current_request and event.quality > state.quality
    drag_preview = (
        state.dragging
        and not current_request
        and event.key != 0
        and (
            event.quality == 1 or (state.scope and (not state.linear))
        )
        and state.quality < 2
    )
    historical_frame = (state.scope and state.history and event.history
                        and not current_request and event.key > 0
                        and event.epoch < state.epoch)
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and state.ready
        and valid_quality
        and event.signal == state.signal
        and scope_compatible
        and (
            settled_progress or drag_preview or historical_frame
        )
    )
    update_error = accepted and current_request and state.scope
    update_quality = accepted and current_request
    return (
        replace(
            state,
            display_available=True if accepted and state.scope else state.display_available,
            display_epoch=event.epoch if accepted and state.scope else state.display_epoch,
            display_key=event.key if accepted and state.scope else state.display_key,
            display_provenance=(
                (
                    event.provenance
                )
                if accepted and state.scope
                else state.display_provenance
            ),
            display_resolution=(
                (
                    event.resolution
                )
                if accepted and state.scope
                else state.display_resolution
            ),
            display_signal=event.signal if accepted and state.scope else state.display_signal,
            error=False if update_error else state.error,
            error_key=0 if update_error else state.error_key,
            publications=state.publications + 1 if update_quality else state.publications,
            quality=event.quality if update_quality else state.quality,
        ),
        Effects(accept=accepted, drop=not accepted, redraw=accepted),
    )


def apply_ready(state: State, event: Event) -> tuple[State, Effects]:
    return (
        replace(
            state,
            ready=(
                (
                    event.ready
                )
                if state.live and event.owner == state.owner and same_session(state, event)
                else state.ready
            ),
        ),
        Effects(),
    )


def apply_recover(state: State, event: Event) -> tuple[State, Effects]:
    update_owner = state.live and event.owner == state.owner and same_session(state, event)
    return (
        replace(
            state,
            display_available=False if update_owner else state.display_available,
            display_epoch=0 if update_owner else state.display_epoch,
            display_key=0 if update_owner else state.display_key,
            display_provenance=0 if update_owner else state.display_provenance,
            display_resolution=0 if update_owner else state.display_resolution,
            display_signal=0 if update_owner else state.display_signal,
            dragging=False if update_owner else state.dragging,
            effect=0 if update_owner else state.effect,
            effect_phase=0 if update_owner else state.effect_phase,
            epoch=state.epoch + 1 if update_owner else state.epoch,
            error=False if update_owner else state.error,
            error_key=0 if update_owner else state.error_key,
            live=True if update_owner else state.live,
            owner=state.owner + 1 if update_owner else state.owner,
            provenance=0 if update_owner else state.provenance,
            publications=0 if update_owner else state.publications,
            quality=0 if update_owner else state.quality,
            ready=True if update_owner else state.ready,
            requested=0 if update_owner else state.requested,
            session=state.session + 1 if update_owner else state.session,
            signal=0 if update_owner else state.signal,
            worker=0 if update_owner else state.worker,
            worker_epoch=0 if update_owner else state.worker_epoch,
            worker_key=0 if update_owner else state.worker_key,
        ),
        Effects(cancel=update_owner and state.worker != 0, recovered=update_owner),
    )


def apply_release(state: State, event: Event) -> tuple[State, Effects]:
    return (
        replace(
            state,
            dragging=(
                (
                    False
                )
                if state.live and event.owner == state.owner and same_session(state, event)
                else state.dragging
            ),
        ),
        Effects(),
    )


def apply_replace(state: State, event: Event) -> tuple[State, Effects]:
    return (
        state,
        Effects(
            cancel=(
                state.live
                and event.owner == state.owner
                and same_session(state, event)
                and event.key != 0
                and (
                    event.key != state.requested or event.cancelled
                )
                and not (
                    (
                        event.started
                        and state.dragging
                        and not event.cancelled
                        and (
                            not event.preview or event.scope_demand
                        )
                        and not event.finished
                    )
                )
            ),
            replace=(
                state.live
                and event.owner == state.owner
                and same_session(state, event)
                and event.key != 0
                and (
                    event.key != state.requested or event.cancelled
                )
                and (
                    (
                        not event.started
                        or (
                            (
                                event.finished
                                and (
                                    (
                                        not state.dragging
                                        or event.published
                                        or event.key == state.requested
                                        or not event.preview
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        ),
    )


def apply_request(state: State, event: Event) -> tuple[State, Effects]:
    update_epoch = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and event.key != 0
        and (
            event.key != state.requested or (state.scope and event.provenance != state.provenance)
        )
    )
    return (
        replace(
            state,
            epoch=state.epoch + 1 if update_epoch else state.epoch,
            error=False if update_epoch else state.error,
            error_key=0 if update_epoch else state.error_key,
            provenance=event.provenance if update_epoch and state.scope else state.provenance,
            publications=0 if update_epoch else state.publications,
            quality=0 if update_epoch else state.quality,
            requested=event.key if update_epoch else state.requested,
            history=(state.scope and event.history) if update_epoch else state.history,
        ),
        Effects(),
    )


def apply_retry(state: State, event: Event) -> tuple[State, Effects]:
    accepted = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and state.scope
        and event.key != 0
        and event.key == state.requested
        and state.worker == 0
        and state.error
        and event.key == state.error_key
    )
    return (
        replace(
            state,
            epoch=state.epoch + 1 if accepted else state.epoch,
            error=False if accepted else state.error,
            error_key=0 if accepted else state.error_key,
            publications=0 if accepted else state.publications,
            quality=0 if accepted else state.quality,
        ),
        Effects(accept=accepted, drop=not accepted),
    )


def apply_shutdown(state: State, event: Event) -> tuple[State, Effects]:
    update_owner = state.live and event.owner == state.owner and same_session(state, event)
    return (
        replace(
            state,
            display_available=False if update_owner else state.display_available,
            display_epoch=0 if update_owner else state.display_epoch,
            display_key=0 if update_owner else state.display_key,
            display_provenance=0 if update_owner else state.display_provenance,
            display_resolution=0 if update_owner else state.display_resolution,
            display_signal=0 if update_owner else state.display_signal,
            dragging=False if update_owner else state.dragging,
            effect=0 if update_owner else state.effect,
            effect_phase=0 if update_owner else state.effect_phase,
            epoch=state.epoch + 1 if update_owner else state.epoch,
            error=False if update_owner else state.error,
            error_key=0 if update_owner else state.error_key,
            live=False if update_owner else state.live,
            owner=state.owner + 1 if update_owner else state.owner,
            provenance=0 if update_owner else state.provenance,
            publications=0 if update_owner else state.publications,
            quality=0 if update_owner else state.quality,
            ready=True if update_owner else state.ready,
            requested=0 if update_owner else state.requested,
            session=state.session + 1 if update_owner else state.session,
            signal=0 if update_owner else state.signal,
            worker=0 if update_owner else state.worker,
            worker_epoch=0 if update_owner else state.worker_epoch,
            worker_key=0 if update_owner else state.worker_key,
        ),
        Effects(cancel=update_owner and state.worker != 0),
    )


def apply_signal(state: State, event: Event) -> tuple[State, Effects]:
    update_display_available = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and state.scope
        and event.signal != state.signal
    )
    return (
        replace(
            state,
            display_available=False if update_display_available else state.display_available,
            display_epoch=0 if update_display_available else state.display_epoch,
            display_key=0 if update_display_available else state.display_key,
            display_provenance=0 if update_display_available else state.display_provenance,
            display_resolution=0 if update_display_available else state.display_resolution,
            display_signal=0 if update_display_available else state.display_signal,
            signal=(
                (
                    event.signal
                )
                if state.live and event.owner == state.owner and same_session(state, event)
                else state.signal
            ),
        ),
        Effects(),
    )


def apply_start(state: State, event: Event) -> tuple[State, Effects]:
    update_worker = (
        state.live
        and event.owner == state.owner
        and same_session(state, event)
        and state.worker == 0
        and event.token != 0
        and event.key != 0
        and (
            (event.key == state.requested and (not state.scope or event.epoch == state.epoch)) or (state.scope and state.dragging)
            or (state.scope and state.history and event.history and event.epoch <= state.epoch)
        )
        and not (state.scope and state.error and (event.key == state.error_key))
        and (
            (
                not state.scope
                or (
                    (
                        event.resolution == 65
                        and event.provenance != 0
                        and (
                            event.key != state.requested or event.provenance == state.provenance
                        )
                    )
                )
            )
        )
    )
    return (
        replace(
            state,
            worker=event.token if update_worker else state.worker,
            worker_epoch=state.epoch if update_worker else state.worker_epoch,
            worker_key=event.key if update_worker else state.worker_key,
        ),
        Effects(drop=not update_worker, start=update_worker),
    )


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'begin':
        return apply_begin(state, event)
    if event.kind == 'complete':
        return apply_complete(state, event)
    if event.kind == 'effect_begin':
        return apply_effect_begin(state, event)
    if event.kind == 'effect_commit':
        return apply_effect_commit(state, event)
    if event.kind == 'effect_fail':
        return apply_effect_fail(state, event)
    if event.kind == 'fail':
        return apply_fail(state, event)
    if event.kind == 'invalidate':
        return apply_invalidate(state, event)
    if event.kind == 'publish':
        return apply_publish(state, event)
    if event.kind == 'ready':
        return apply_ready(state, event)
    if event.kind == 'recover':
        return apply_recover(state, event)
    if event.kind == 'release':
        return apply_release(state, event)
    if event.kind == 'replace':
        return apply_replace(state, event)
    if event.kind == 'request':
        return apply_request(state, event)
    if event.kind == 'retry':
        return apply_retry(state, event)
    if event.kind == 'shutdown':
        return apply_shutdown(state, event)
    if event.kind == 'signal':
        return apply_signal(state, event)
    if event.kind == 'start':
        return apply_start(state, event)
    raise ValueError('Unknown machine event')


def _final_scope_resolution_all_resolution(_items: tuple[int, ...]) -> bool:
    _result = True
    for resolution in _items:
        _result = _result and resolution == 65
    return _result


def _cyclic_first(values: tuple[bool, ...], start: int) -> int:
    count = len(values)
    if count == 0:
        return -1
    cursor = start % count
    for offset in range(count):
        index = (cursor + offset) % count
        if values[index]:
            return index
    return -1
