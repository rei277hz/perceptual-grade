"""Bounded manual bake admission, worker leases and destination ordering.

Payloads remain in the Blender adapter. Frozen and terminal/retirement receipts
must identify this exact entry; logical cancellation never frees physical work.
"""
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class Entry:
    request: int = 0
    target: int = 0
    destination: int = 0
    phase: Literal['empty', 'freezing', 'queued', 'running', 'ready', 'publishing', 'cancelled'] = 'empty'
    lane: int = 0
    physical: bool = False


@dataclass(frozen=True)
class State:
    generation: int = 1
    serial: int = 0
    closing: bool = False
    entries: tuple[Entry, Entry, Entry, Entry, Entry, Entry, Entry, Entry] = (Entry(), Entry(), Entry(), Entry(), Entry(), Entry(), Entry(), Entry())


@dataclass(frozen=True)
class Event:
    kind: Literal['admit', 'frozen', 'start', 'complete', 'publish', 'cancel', 'retired', 'finish', 'close']
    generation: int = 1
    slot: int = 0
    request: int = 0
    target: int = 0
    destination: int = 0
    lane: int = 0
    validated: bool = False


@dataclass(frozen=True)
class Effects:
    accepted: bool = False
    action: Literal['none', 'freeze', 'queue', 'dispatch', 'stage', 'publish', 'cancel', 'retire', 'release', 'cancel_all'] = 'none'
    lane: int = 0


def new() -> State:
    return State()


def put(state: State, slot: int, value: Entry) -> State:
    return replace(state, entries=(
        value if slot == 0 else state.entries[0], value if slot == 1 else state.entries[1], value if slot == 2 else state.entries[2], value if slot == 3 else state.entries[3], value if slot == 4 else state.entries[4], value if slot == 5 else state.entries[5], value if slot == 6 else state.entries[6], value if slot == 7 else state.entries[7]))


def target_busy(state: State, target: int) -> bool:
    return target > 0 and (state.entries[0].target == target or state.entries[1].target == target or state.entries[2].target == target or state.entries[3].target == target or state.entries[4].target == target or state.entries[5].target == target or state.entries[6].target == target or state.entries[7].target == target)


def lane_busy(state: State, lane: int) -> bool:
    return lane > 0 and (state.entries[0].lane == lane or state.entries[1].lane == lane or state.entries[2].lane == lane or state.entries[3].lane == lane or state.entries[4].lane == lane or state.entries[5].lane == lane or state.entries[6].lane == lane or state.entries[7].lane == lane)


def earlier(entry: Entry, request: int, destination: int) -> bool:
    return entry.request > 0 and entry.request < request and entry.destination == destination


def destination_waits(state: State, request: int, destination: int) -> bool:
    return earlier(state.entries[0], request, destination) or earlier(state.entries[1], request, destination) or earlier(state.entries[2], request, destination) or earlier(state.entries[3], request, destination) or earlier(state.entries[4], request, destination) or earlier(state.entries[5], request, destination) or earlier(state.entries[6], request, destination) or earlier(state.entries[7], request, destination)


def cancel_entry(entry: Entry) -> Entry:
    return replace(entry, phase='cancelled') if entry.request > 0 else entry


def step(state: State, event: Event) -> tuple[State, Effects]:
    if event.generation != state.generation:
        return state, Effects()
    if event.kind == 'close':
        return replace(state, closing=True, entries=(
            cancel_entry(state.entries[0]), cancel_entry(state.entries[1]), cancel_entry(state.entries[2]), cancel_entry(state.entries[3]), cancel_entry(state.entries[4]), cancel_entry(state.entries[5]), cancel_entry(state.entries[6]), cancel_entry(state.entries[7]))), Effects(True, 'cancel_all')
    if not 0 <= event.slot < 8:
        return state, Effects()
    entry = state.entries[event.slot]
    if event.kind == 'admit':
        if (not state.closing and entry.request == 0 and event.request > state.serial
                and event.target > 0 and event.destination > 0 and not target_busy(state, event.target)):
            value = Entry(event.request, event.target, event.destination, 'freezing')
            return put(replace(state, serial=event.request), event.slot, value), Effects(True, 'freeze')
        return state, Effects()
    if (entry.request == 0 or event.request != entry.request or event.target != entry.target
            or event.destination != entry.destination):
        return state, Effects()
    if event.kind == 'cancel':
        return put(state, event.slot, cancel_entry(entry)), Effects(True, 'cancel' if entry.physical else 'none')
    if event.kind == 'retired' and entry.physical and event.validated:
        return put(state, event.slot, replace(entry, physical=False, lane=0)), Effects(True, 'retire', entry.lane)
    if event.kind == 'finish' and entry.phase in ('publishing', 'cancelled') and not entry.physical and event.validated:
        return put(state, event.slot, Entry()), Effects(True, 'release')
    if state.closing:
        return state, Effects()
    if event.kind == 'frozen' and entry.phase == 'freezing' and event.validated:
        return put(state, event.slot, replace(entry, phase='queued')), Effects(True, 'queue')
    if (event.kind == 'start' and entry.phase == 'queued' and event.validated
            and event.lane in (1, 2) and not lane_busy(state, event.lane)
            and not destination_waits(state, entry.request, entry.destination)):
        return put(state, event.slot, replace(entry, phase='running', physical=True, lane=event.lane)), Effects(True, 'dispatch', event.lane)
    if event.kind == 'complete' and entry.phase == 'running' and entry.physical and event.validated:
        return put(state, event.slot, replace(entry, phase='ready', physical=False, lane=0)), Effects(True, 'stage', entry.lane)
    if (event.kind == 'publish' and entry.phase == 'ready' and event.validated
            and not destination_waits(state, entry.request, entry.destination)):
        return put(state, event.slot, replace(entry, phase='publishing')), Effects(True, 'publish')
    return state, Effects()
