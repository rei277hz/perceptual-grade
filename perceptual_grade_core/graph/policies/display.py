"""Scope display identity, visual readiness, and matching statistics."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class ScopeDisplayEffects:
    accept: bool = False
    replace_visual: bool = False
    replace_statistics: bool = False
    clear: bool = False


@dataclass(frozen=True)
class ScopeDisplayEvent:
    kind: Literal[
        'request',
        'visual',
        'statistics',
        'invalidate',
        'release',
        'cancel',
        'failure',
        'retry',
        'reset',
        'geometry',
    ]
    source: int = 0
    slot: int = 0
    analysis: int = 0
    geometry: int = 0
    epoch: int = 0
    capture: int = 0
    validated: bool = False


@dataclass(frozen=True)
class ScopeDisplayState:
    source: int = 0
    slot: int = 0
    desired_analysis: int = 0
    desired_geometry: int = 0
    desired_epoch: int = 0
    visual_available: bool = False
    visual_epoch: int = 0
    visual_analysis: int = 0
    visual_capture: int = 0
    visual_geometry: int = 0
    stats_available: bool = False
    stats_epoch: int = 0
    stats_analysis: int = 0
    stats_capture: int = 0


def display_domain(event: ScopeDisplayEvent) -> bool:
    return (
        event.source >= 0
        and event.slot >= 0
        and event.analysis >= 0
        and event.geometry >= 0
        and event.epoch >= 0
        and event.capture >= 0
    )


def display_invariant(state: ScopeDisplayState) -> bool:
    return (
        state.source >= 0
        and state.slot >= 0
        and state.desired_analysis >= 0
        and state.desired_geometry >= 0
        and state.desired_epoch >= 0
        and state.visual_epoch >= 0
        and state.visual_analysis >= 0
        and state.visual_capture >= 0
        and state.visual_geometry >= 0
        and state.stats_epoch >= 0
        and state.stats_analysis >= 0
        and state.stats_capture >= 0
        and state.visual_epoch <= state.desired_epoch
        and state.stats_epoch <= state.desired_epoch
        and (
            (
                not state.desired_epoch > 0
                or (
                    state.source > 0 and state.slot > 0 and (state.desired_analysis > 0)
                )
            )
        )
        and (
            (
                not state.visual_available
                or (
                    (
                        state.source > 0
                        and state.slot > 0
                        and state.visual_epoch > 0
                        and state.visual_analysis > 0
                        and state.visual_capture > 0
                    )
                )
            )
        )
        and (
            (
                not state.stats_available
                or (
                    (
                        state.source > 0
                        and state.slot > 0
                        and state.stats_epoch > 0
                        and state.stats_analysis > 0
                        and state.stats_capture > 0
                    )
                )
            )
        )
        and (
            (
                not not state.visual_available
                or (
                    (
                        state.visual_epoch == 0
                        and state.visual_analysis == 0
                        and state.visual_capture == 0
                    )
                )
            )
        )
        and (
            (
                not not state.stats_available
                or (
                    (
                        state.stats_epoch == 0
                        and state.stats_analysis == 0
                        and state.stats_capture == 0
                    )
                )
            )
        )
    )


def _domain_event(state: ScopeDisplayState, event: ScopeDisplayEvent) -> bool:
    return display_domain(event)


def new() -> ScopeDisplayState:
    return ScopeDisplayState()


def scope_display_current(state: ScopeDisplayState, statistics_required: bool) -> bool:
    return (
        state.visual_available
        and state.visual_analysis == state.desired_analysis
        and state.visual_geometry == state.desired_geometry
        and (
            (
                not statistics_required
                or (
                    state.stats_available and state.stats_analysis == state.desired_analysis
                )
            )
        )
    )


def scope_display_statistics_match(state: ScopeDisplayState) -> bool:
    return (
        state.visual_available
        and state.stats_available
        and state.visual_analysis == state.stats_analysis
    )


def _domain_state(state: ScopeDisplayState) -> bool:
    return display_invariant(state)


def state_valid(state: ScopeDisplayState) -> bool:
    return _domain_state(state)


def event_valid(state: ScopeDisplayState, event: ScopeDisplayEvent) -> bool:
    return _domain_event(state, event)


def apply_cancel(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_failure(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_geometry(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_invalidate(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (
        replace(
            state,
            desired_analysis=0,
            desired_epoch=0,
            desired_geometry=0,
            slot=event.slot,
            source=event.source,
            stats_analysis=0,
            stats_available=False,
            stats_capture=0,
            stats_epoch=0,
            visual_analysis=0,
            visual_available=False,
            visual_capture=0,
            visual_epoch=0,
            visual_geometry=0,
        ),
        ScopeDisplayEffects(accept=True, clear=True),
    )


def apply_release(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_request(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    accepted = (
        state.source > 0
        and state.slot > 0
        and event.source == state.source
        and event.slot == state.slot
        and event.analysis > 0
        and event.epoch > 0
        and event.epoch >= state.desired_epoch
    )
    return (
        replace(
            state,
            desired_analysis=event.analysis if accepted else state.desired_analysis,
            desired_epoch=event.epoch if accepted else state.desired_epoch,
            desired_geometry=event.geometry if accepted else state.desired_geometry,
        ),
        ScopeDisplayEffects(accept=accepted),
    )


def apply_reset(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_retry(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    return (state, ScopeDisplayEffects())


def apply_statistics(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    accepted = (
        state.source > 0
        and state.slot > 0
        and event.source == state.source
        and event.slot == state.slot
        and event.validated
        and event.analysis > 0
        and event.capture > 0
        and event.epoch > 0
        and event.epoch <= state.desired_epoch
        and (
            event.epoch < state.desired_epoch or event.analysis == state.desired_analysis
        )
        and event.epoch >= state.stats_epoch
        and (
            event.epoch > state.stats_epoch or event.analysis == state.stats_analysis
        )
    )
    return (
        replace(
            state,
            stats_analysis=event.analysis if accepted else state.stats_analysis,
            stats_available=True if accepted else state.stats_available,
            stats_capture=event.capture if accepted else state.stats_capture,
            stats_epoch=event.epoch if accepted else state.stats_epoch,
        ),
        ScopeDisplayEffects(accept=accepted, replace_statistics=accepted),
    )


def apply_visual(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    accepted = (
        state.source > 0
        and state.slot > 0
        and event.source == state.source
        and event.slot == state.slot
        and event.validated
        and event.analysis > 0
        and event.capture > 0
        and event.epoch > 0
        and event.epoch <= state.desired_epoch
        and (
            event.epoch < state.desired_epoch or event.analysis == state.desired_analysis
        )
        and event.epoch >= state.visual_epoch
        and (
            event.epoch > state.visual_epoch or event.analysis == state.visual_analysis
        )
    )
    return (
        replace(
            state,
            visual_analysis=event.analysis if accepted else state.visual_analysis,
            visual_available=True if accepted else state.visual_available,
            visual_capture=event.capture if accepted else state.visual_capture,
            visual_epoch=event.epoch if accepted else state.visual_epoch,
            visual_geometry=event.geometry if accepted else state.visual_geometry,
        ),
        ScopeDisplayEffects(accept=accepted, replace_visual=accepted),
    )


def step(
    state: ScopeDisplayState,
    event: ScopeDisplayEvent,
) -> tuple[ScopeDisplayState, ScopeDisplayEffects]:
    if event.kind == 'cancel':
        return apply_cancel(state, event)
    if event.kind == 'failure':
        return apply_failure(state, event)
    if event.kind == 'geometry':
        return apply_geometry(state, event)
    if event.kind == 'invalidate':
        return apply_invalidate(state, event)
    if event.kind == 'release':
        return apply_release(state, event)
    if event.kind == 'request':
        return apply_request(state, event)
    if event.kind == 'reset':
        return apply_reset(state, event)
    if event.kind == 'retry':
        return apply_retry(state, event)
    if event.kind == 'statistics':
        return apply_statistics(state, event)
    if event.kind == 'visual':
        return apply_visual(state, event)
    raise ValueError('Unknown machine event')
