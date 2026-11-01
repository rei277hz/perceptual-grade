"""Immutable ownership edges and reference counts for native effects."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class EffectState:
    occurrence: int
    before: int = 0
    after: int = 0
    current: int = 0
    known: bool = True
    armed: bool = False
    applied: bool = False
    failed: bool = False
    retired: bool = False


@dataclass(frozen=True)
class Effects:
    pass


@dataclass(frozen=True)
class Event:
    kind: Literal['ARM', 'APPLY', 'COMPENSATED', 'COMPENSATION_FAILED', 'COMMIT']
    occurrence: int


def edge_count(edges: tuple[int | str, ...], selected: int | str) -> int:
    return _edge_count_count_edge(edges, selected)


def new(occurrence: int, before: int, after: int) -> EffectState:
    if not (occurrence > 0 and before >= 0 and (after >= 0)):
        raise ValueError('Invalid domain for new')
    return EffectState(after=after, before=before, current=before, occurrence=occurrence)


def state_valid(state: EffectState) -> bool:
    return True


def event_valid(state: EffectState, event: Event) -> bool:
    return True


def apply_APPLY(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    update_current = event.occurrence == state.occurrence and (not state.retired) and state.armed
    return (
        replace(
            state,
            applied=True if update_current else state.applied,
            current=state.after if update_current else state.current,
            known=True if update_current else state.known,
        ),
        Effects(),
    )


def apply_ARM(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    update_known = (
        event.occurrence == state.occurrence
        and not state.retired
        and not state.armed
    )
    return (
        replace(
            state,
            armed=True if update_known else state.armed,
            known=False if update_known else state.known,
        ),
        Effects(),
    )


def apply_COMMIT(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    update_armed = (
        event.occurrence == state.occurrence
        and not state.retired
        and state.applied
        and state.known
        and not state.failed
    )
    return (
        replace(
            state,
            armed=False if update_armed else state.armed,
            retired=True if update_armed else state.retired,
        ),
        Effects(),
    )


def apply_COMPENSATED(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    update_current = event.occurrence == state.occurrence and (not state.retired) and state.armed
    return (
        replace(
            state,
            armed=False if update_current else state.armed,
            current=state.before if update_current else state.current,
            failed=False if update_current else state.failed,
            known=True if update_current else state.known,
            retired=True if update_current else state.retired,
        ),
        Effects(),
    )


def apply_COMPENSATION_FAILED(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    update_known = event.occurrence == state.occurrence and (not state.retired) and state.armed
    return (
        replace(
            state,
            failed=True if update_known else state.failed,
            known=False if update_known else state.known,
        ),
        Effects(),
    )


def step(state: EffectState, event: Event) -> tuple[EffectState, Effects]:
    if event.kind == 'APPLY':
        return apply_APPLY(state, event)
    if event.kind == 'ARM':
        return apply_ARM(state, event)
    if event.kind == 'COMMIT':
        return apply_COMMIT(state, event)
    if event.kind == 'COMPENSATED':
        return apply_COMPENSATED(state, event)
    if event.kind == 'COMPENSATION_FAILED':
        return apply_COMPENSATION_FAILED(state, event)
    raise ValueError('Unknown machine event')


def _edge_count_count_edge(_items: tuple[int | str, ...], selected: int | str) -> int:
    _result = 0
    for edge in _items:
        _result = _result + (1 if edge == selected else 0)
    return _result
