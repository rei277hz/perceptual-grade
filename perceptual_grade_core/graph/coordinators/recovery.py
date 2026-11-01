"""Bounded recovery of native output and explicitly saved control wrappers.

The adapter freezes a finite owner worklist after physical retirement. Each
validated receipt settles exactly its next item. Adoption records observations;
it cannot certify numerical output. A separate wrapper-restoration receipt is
required for every adopted owner before runtime activation.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class State:
    generation: int = 0
    phase: Literal['idle','replacing','retiring','discovering','adopting','restoring','reconciling','completed','failed','superseded'] = 'idle'
    total: int = 0
    settled: int = 0
    restored: int = 0
    physical: bool = False
    replaced: bool = False


@dataclass(frozen=True)
class Event:
    kind: Literal['begin','replaced','retired','discovered','adopted','wrapper_restored','reconciled','failed','supersede']
    generation: int = 0
    native: bool = False
    validated: bool = False
    workers: int = 0
    total: int = 0
    index: int = 0


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none','fence','retire_journals','discover','adopt','restore_wrapper','reconcile','activate','fail','supersede'] = 'none'
    certified: bool = False


def new() -> State:
    return State()


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.kind == 'begin':
        if event.generation > state.generation and state.phase in ('idle','completed','failed','superseded'):
            return State(event.generation,'replacing' if event.native else 'retiring',physical=True),Effects(True,'fence')
        return state,Effects()
    if event.generation != state.generation:
        return state,Effects()
    if event.kind == 'supersede' and state.phase not in ('idle','superseded'):
        return replace(state,phase='superseded'),Effects(True,'supersede')
    if state.phase in ('idle','completed','failed','superseded'):
        return state,Effects()
    if event.kind == 'failed':
        return replace(state,phase='failed'),Effects(True,'fail')
    if event.kind == 'replaced' and state.phase == 'replacing' and event.native and event.validated:
        return replace(state,phase='retiring',replaced=True),Effects(True,'retire_journals')
    if event.kind == 'retired' and state.phase == 'retiring' and event.validated and event.workers == 0:
        return replace(state,phase='discovering',physical=False),Effects(True,'discover')
    if event.kind == 'discovered' and state.phase == 'discovering' and event.validated and event.total >= 1:
        return replace(state,total=event.total,phase='adopting' if event.total > 1 else 'reconciling'),Effects(True,'adopt' if event.total > 1 else 'reconcile')
    if event.kind == 'adopted' and state.phase == 'adopting' and event.validated and event.index == state.settled:
        finished = state.settled + 1 == state.total - 1
        return replace(state,settled=state.settled+1,phase='restoring' if finished else 'adopting'),Effects(True,'restore_wrapper' if finished else 'adopt')
    if event.kind == 'wrapper_restored' and state.phase == 'restoring' and event.validated and event.index == state.restored:
        restored = state.restored + 1
        finished = restored == state.total - 1
        return replace(state,restored=restored,phase='reconciling' if finished else 'restoring'),Effects(True,'reconcile' if finished else 'restore_wrapper')
    if event.kind == 'reconciled' and state.phase == 'reconciling' and event.validated and state.restored == state.total - 1:
        return replace(state,settled=state.total,phase='completed'),Effects(True,'activate')
    return state,Effects()
