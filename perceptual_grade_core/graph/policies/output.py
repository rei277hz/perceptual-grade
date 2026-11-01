"""Output-slot authority, transfer, graph selection, and packing state."""
from dataclasses import dataclass, replace


from typing import Literal


@dataclass(frozen=True)
class OutputEffects:
    accepted: bool = False


@dataclass(frozen=True)
class OutputEvent:
    kind: Literal[
        'reserve',
        'armed',
        'transferred',
        'fail',
        'inspect',
        'certify',
        'repair',
        'graph_begin',
        'fallback',
        'rows',
        'select',
        'graph_fail',
        'graph_repair',
        'adopt',
        'uncertify',
        'fallback_done',
        'invalidate',
        'observe_packing',
        'packed',
        'pack_failed',
        'adopt_native',
        'intent',
        'protect',
    ]
    slot: int = -1
    incarnation: int = 0
    source: int | str = 0
    operation: int = 0
    validated: bool = False
    reader_excluded: bool = False
    session: int | str = 0
    output_key: int | str = 0
    readable: bool = False
    certified: bool = False
    layout_valid: bool = True
    clean: bool = False
    intent: int | str = 0
    explicit: bool = False


@dataclass(frozen=True)
class Slot:
    valid: bool = False
    source: int | str = 0
    generation: int = 0
    debt: bool = False
    readability: int = 1
    layout_valid: bool = True
    packing: int = 0


@dataclass(frozen=True)
class State:
    incarnation: int = 1
    selected: int = -1
    pending: int = -1
    operation: int = 0
    pending_source: int | str = 0
    slots: tuple[Slot, Slot, Slot, Slot] = (Slot(), Slot(), Slot(), Slot())
    session: int | str = 0
    write_phase: int = 0
    graph_operation: int = 0
    graph_slot: int = -1
    graph_source: int | str = 0
    graph_output: int | str = 0
    graph_phase: int = 0
    graph_valid: bool = False
    graph_previous_valid: bool = False
    graph_debt: bool = False
    output_key: int | str = 0
    output_source: int | str = 0
    adopted: bool = False
    intent: int | str = 0
    protected: int = -1


def armed_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, debt=True, packing=2, readability=0) if accepted and targeted else slot


def certify_slot(
    slot: Slot,
    targeted: bool,
    accepted: bool,
    state: State,
    event: OutputEvent,
) -> Slot:
    return (
        (
            replace(slot, debt=False, generation=state.incarnation, source=event.source, valid=True)
        )
        if accepted and targeted
        else slot
    )


def complete(state: State, index: int, source: int | str) -> bool:
    return (
        (
            index == 0 and slot_complete(_at(state.slots, 0), state.incarnation, source)
        )
        or (
            index == 1 and slot_complete(_at(state.slots, 1), state.incarnation, source)
        )
        or (
            index == 2 and slot_complete(_at(state.slots, 2), state.incarnation, source)
        )
        or (
            index == 3 and slot_complete(_at(state.slots, 3), state.incarnation, source)
        )
    )


def establish(incarnation: int, uncertain: bool, session: int | str) -> State:
    if not incarnation > 0:
        raise ValueError('Invalid domain for establish')
    return State(
        incarnation=incarnation,
        session=session,
        slots=(
            Slot(debt=uncertain, readability=0 if uncertain else 1),
            Slot(debt=uncertain, readability=0 if uncertain else 1),
            Slot(debt=uncertain, readability=0 if uncertain else 1),
            Slot(debt=uncertain, readability=0 if uncertain else 1),
        ),
    )


def _domain_event(state: State, event: OutputEvent) -> bool:
    return event.incarnation >= 1 and event.operation >= 0


def fail_slot(slot: Slot, targeted: bool, accepted: bool, state: State) -> Slot:
    affected = accepted and targeted and (state.write_phase != 3)
    return replace(slot, debt=True, readability=0, valid=False) if affected else slot


def inspect_slot(slot: Slot, targeted: bool, accepted: bool, event: OutputEvent) -> Slot:
    affected = accepted and targeted and (not (event.readable and event.layout_valid))
    return replace(
        slot,
        debt=True if affected else False if accepted and targeted else slot.debt,
        layout_valid=event.layout_valid if accepted and targeted else slot.layout_valid,
        packing=0 if affected else slot.packing,
        readability=(
            (
                1 if event.readable and event.layout_valid else 2
            )
            if accepted and targeted
            else slot.readability
        ),
        valid=False if affected else slot.valid,
    )


def invalidate_slot(slot: Slot) -> Slot:
    return replace(slot, debt=True, packing=0, readability=0, valid=False)


def observe_packing_slot(slot: Slot, targeted: bool, accepted: bool, event: OutputEvent) -> Slot:
    return replace(slot, packing=1 if event.clean else 2) if accepted and targeted else slot


def output_complete(state: State, output_key: int | str) -> bool:
    return (
        state.graph_valid
        and state.output_key == output_key
        and complete(state, state.selected, state.output_source)
    )


def pack_failed_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, packing=2) if accepted and targeted else slot


def packed_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, packing=1) if accepted and targeted else slot


def packing_clean(state: State, index: int) -> bool:
    return (
        (
            index == 0 and _at(state.slots, 0).packing == 1
        )
        or (
            index == 1 and _at(state.slots, 1).packing == 1
        )
        or (
            index == 2 and _at(state.slots, 2).packing == 1
        )
        or (
            index == 3 and _at(state.slots, 3).packing == 1
        )
    )


def repair_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, debt=False, readability=1, valid=False) if accepted and targeted else slot


def reserve_slot(
    slot: Slot,
    targeted: bool,
    accepted: bool,
    state: State,
    event: OutputEvent,
) -> Slot:
    return (
        (
            replace(
                slot,
                debt=True,
                generation=state.incarnation,
                packing=2,
                source=event.source,
                valid=False,
            )
        )
        if accepted and targeted
        else slot
    )


def same_incarnation(state: State, event: OutputEvent) -> bool:
    return event.incarnation == state.incarnation


def same_session(state: State, event: OutputEvent) -> bool:
    return event.session == state.session


def slot_complete(slot: Slot, incarnation: int, source: int | str) -> bool:
    return (
        slot.valid
        and not slot.debt
        and slot.readability == 1
        and slot.layout_valid
        and slot.generation == incarnation
        and slot.source == source
    )


def slot_consistent(state: State, index: int) -> bool:
    if not (index >= 0 and index < 4):
        raise ValueError('Invalid domain for slot_consistent')
    slot = _at(state.slots, index)
    return (
        slot.generation >= 0
        and slot.generation <= state.incarnation
        and slot.readability >= 0
        and slot.readability <= 3
        and slot.packing >= 0
        and slot.packing <= 2
        and (
            (
                not slot.valid
                or (
                    (
                        not slot.debt
                        and slot.layout_valid
                        and slot.readability == 1
                        and slot.generation == state.incarnation
                    )
                )
            )
        )
        and (
            not not slot.debt or slot.readability == 1 or slot.readability == 3
        )
        and (
            not state.selected == index or slot.valid
        )
        and (
            (
                not state.pending == index
                or (
                    (
                        slot.generation == state.incarnation
                        and slot.source == state.pending_source
                        and slot.layout_valid
                    )
                )
            )
        )
        and (
            not (state.pending == index and state.write_phase == 3) or slot.valid
        )
        and (
            (
                not (state.pending == index and state.write_phase != 3)
                or (
                    not slot.valid and slot.debt
                )
            )
        )
    )


def slot_for(state: State, index: int) -> Slot:
    if not (index >= 0 and index < 4):
        raise ValueError('Invalid domain for slot_for')
    return _at(state.slots, index)


def slot_layout(state: State, index: int) -> bool:
    return (
        (
            index == 0 and _at(state.slots, 0).layout_valid
        )
        or (
            index == 1 and _at(state.slots, 1).layout_valid
        )
        or (
            index == 2 and _at(state.slots, 2).layout_valid
        )
        or (
            index == 3 and _at(state.slots, 3).layout_valid
        )
    )


def slot_readable(state: State, index: int) -> bool:
    return (
        (
            (
                index == 0
                and _at(state.slots, 0).readability == 1
                and _at(state.slots, 0).layout_valid
                and not _at(state.slots, 0).debt
            )
        )
        or (
            (
                index == 1
                and _at(state.slots, 1).readability == 1
                and _at(state.slots, 1).layout_valid
                and not _at(state.slots, 1).debt
            )
        )
        or (
            (
                index == 2
                and _at(state.slots, 2).readability == 1
                and _at(state.slots, 2).layout_valid
                and not _at(state.slots, 2).debt
            )
        )
        or (
            (
                index == 3
                and _at(state.slots, 3).readability == 1
                and _at(state.slots, 3).layout_valid
                and not _at(state.slots, 3).debt
            )
        )
    )


def _domain_state(state: State) -> bool:
    return valid(state)


def transferred_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, debt=False, readability=1, valid=True) if accepted and targeted else slot


def uncertify_slot(slot: Slot, targeted: bool, accepted: bool) -> Slot:
    return replace(slot, valid=False) if accepted and targeted else slot


def valid(state: State) -> bool:
    return (
        state.incarnation >= 1
        and state.operation >= 0
        and state.graph_operation >= 0
        and state.selected >= -1
        and state.selected < 4
        and state.protected >= -1
        and state.protected < 4
        and (not state.adopted or (state.selected == -1 and state.pending == -1
             and not state.graph_valid and state.graph_phase == 0))
        and state.pending >= -1
        and state.pending < 4
        and state.graph_slot >= -1
        and state.graph_slot < 4
        and state.write_phase >= 0
        and state.write_phase <= 3
        and state.graph_phase >= 0
        and state.graph_phase <= 4
        and (state.pending == -1) == (state.write_phase == 0)
        and (
            state.pending == -1 or state.pending != state.selected
        )
        and (
            not state.pending >= 0 or state.operation > 0
        )
        and (
            not state.graph_phase == 0 or state.graph_slot == -1
        )
        and (
            (
                not state.graph_phase != 0
                or (
                    (
                        state.graph_operation > 0
                        and state.graph_debt
                        and (
                            (
                                state.graph_slot == -1
                                or complete(state, state.graph_slot, state.graph_source)
                            )
                        )
                        and (
                            (
                                state.pending == -1
                                or (
                                    state.pending == state.graph_slot and state.write_phase == 3
                                )
                            )
                        )
                        and (
                            (
                                not state.graph_previous_valid
                                or complete(state, state.selected, state.output_source)
                            )
                        )
                    )
                )
            )
        )
        and (
            not state.graph_valid or complete(state, state.selected, state.output_source)
        )
        and slot_consistent(state, 0)
        and slot_consistent(state, 1)
        and slot_consistent(state, 2)
        and slot_consistent(state, 3)
    )


def state_valid(state: State) -> bool:
    return _domain_state(state)


def event_valid(state: State, event: OutputEvent) -> bool:
    return _domain_event(state, event)


def apply_adopt(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and state.pending == -1
        and event.validated
        and (
            state.graph_phase == 0 or state.graph_phase == 4
        )
        and complete(state, event.slot, event.source)
        and packing_clean(state, event.slot)
    )
    return (
        replace(
            state,
            graph_debt=False if accepted else state.graph_debt,
            graph_operation=state.graph_operation + 1 if accepted else state.graph_operation,
            graph_phase=0 if accepted else state.graph_phase,
            graph_slot=-1 if accepted else state.graph_slot,
            graph_valid=True if accepted else state.graph_valid,
            output_key=event.output_key if accepted else state.output_key,
            output_source=event.source if accepted else state.output_source,
            selected=event.slot if accepted else state.selected,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_armed(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot == state.pending
        and event.operation == state.operation
        and event.source == state.pending_source
        and state.write_phase == 1
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(
            state,
            slots=(
                armed_slot(_at(state.slots, 0), event.slot == 0, accepted),
                armed_slot(_at(state.slots, 1), event.slot == 1, accepted),
                armed_slot(_at(state.slots, 2), event.slot == 2, accepted),
                armed_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
            write_phase=2 if accepted else state.write_phase,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_certify(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot != state.pending
        and state.graph_phase == 0
        and slot_readable(state, event.slot)
        and event.validated
        and event.certified
    )
    return (
        replace(
            state,
            graph_valid=(
                (
                    (
                        (
                            False
                        )
                        if event.slot == state.selected and event.source != state.output_source
                        else state.graph_valid
                    )
                )
                if accepted
                else state.graph_valid
            ),
            slots=(
                certify_slot(_at(state.slots, 0), event.slot == 0, accepted, state, event),
                certify_slot(_at(state.slots, 1), event.slot == 1, accepted, state, event),
                certify_slot(_at(state.slots, 2), event.slot == 2, accepted, state, event),
                certify_slot(_at(state.slots, 3), event.slot == 3, accepted, state, event),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_fail(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot == state.pending
        and event.operation == state.operation
        and event.source == state.pending_source
    )
    return (
        replace(
            state,
            pending=-1 if accepted else state.pending,
            slots=(
                fail_slot(_at(state.slots, 0), event.slot == 0, accepted, state),
                fail_slot(_at(state.slots, 1), event.slot == 1, accepted, state),
                fail_slot(_at(state.slots, 2), event.slot == 2, accepted, state),
                fail_slot(_at(state.slots, 3), event.slot == 3, accepted, state),
            ),
            write_phase=0 if accepted else state.write_phase,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_fallback(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
        and state.graph_phase == 1
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(
            state,
            graph_phase=2 if accepted else state.graph_phase,
            graph_valid=False if accepted else state.graph_valid,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_fallback_done(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
        and event.slot == -1
        and state.graph_phase == 3
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(
            state,
            graph_debt=False if accepted else state.graph_debt,
            graph_phase=0 if accepted else state.graph_phase,
            graph_slot=-1 if accepted else state.graph_slot,
            graph_valid=False if accepted else state.graph_valid,
            selected=-1 if accepted else state.selected,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_graph_begin(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and state.graph_phase == 0
        and event.reader_excluded
        and (
            (
                event.slot == -1
                or (
                    complete(state, event.slot, event.source) and packing_clean(state, event.slot)
                )
            )
        )
        and (
            state.pending == -1 or (state.pending == event.slot and state.write_phase == 3)
        )
    )
    return (
        replace(
            state,
            graph_debt=True if accepted else state.graph_debt,
            graph_operation=state.graph_operation + 1 if accepted else state.graph_operation,
            graph_output=event.output_key if accepted else state.graph_output,
            graph_phase=1 if accepted else state.graph_phase,
            graph_previous_valid=state.graph_valid if accepted else state.graph_previous_valid,
            graph_slot=event.slot if accepted else state.graph_slot,
            graph_source=event.source if accepted else state.graph_source,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_graph_fail(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
    )
    return (
        replace(
            state,
            graph_debt=(
                (
                    not (event.validated and event.reader_excluded)
                )
                if accepted
                else state.graph_debt
            ),
            graph_phase=(
                (
                    0 if event.validated and event.reader_excluded else 4
                )
                if accepted
                else state.graph_phase
            ),
            graph_slot=(
                (
                    -1 if event.validated and event.reader_excluded else state.graph_slot
                )
                if accepted
                else state.graph_slot
            ),
            graph_valid=(
                (
                    (
                        (
                            state.graph_previous_valid
                        )
                        if event.validated and event.reader_excluded
                        else False
                    )
                )
                if accepted
                else state.graph_valid
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_graph_repair(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
        and state.graph_phase == 4
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(
            state,
            graph_debt=False if accepted else state.graph_debt,
            graph_phase=0 if accepted else state.graph_phase,
            graph_slot=-1 if accepted else state.graph_slot,
            graph_valid=state.graph_previous_valid if accepted else state.graph_valid,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_inspect(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot != state.pending
        and state.graph_phase == 0
    )
    update_selected = not (event.readable and event.layout_valid) and event.slot == state.selected
    return (
        replace(
            state,
            graph_valid=(
                (
                    False if update_selected else state.graph_valid
                )
                if accepted
                else state.graph_valid
            ),
            selected=(-1 if update_selected else state.selected) if accepted else state.selected,
            slots=(
                inspect_slot(_at(state.slots, 0), event.slot == 0, accepted, event),
                inspect_slot(_at(state.slots, 1), event.slot == 1, accepted, event),
                inspect_slot(_at(state.slots, 2), event.slot == 2, accepted, event),
                inspect_slot(_at(state.slots, 3), event.slot == 3, accepted, event),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_invalidate(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    return (
        replace(
            state,
            graph_debt=False,
            graph_phase=0,
            graph_previous_valid=False,
            graph_slot=-1,
            graph_valid=False,
            incarnation=state.incarnation + 1,
            pending=-1,
            selected=-1,
            slots=(
                invalidate_slot(_at(state.slots, 0)),
                invalidate_slot(_at(state.slots, 1)),
                invalidate_slot(_at(state.slots, 2)),
                invalidate_slot(_at(state.slots, 3)),
            ),
            write_phase=0,
        ),
        OutputEffects(accepted=True),
    )


def apply_observe_packing(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.operation == state.operation
        and (
            (
                state.pending == -1
                or (
                    (
                        same_incarnation(state, event)
                        and same_session(state, event)
                        and event.slot >= 0
                        and event.slot < 4
                        and event.slot == state.pending
                        and event.operation == state.operation
                        and event.source == state.pending_source
                    )
                )
            )
        )
        and state.pending == -1
        and slot_readable(state, event.slot)
        and event.reader_excluded
    )
    return (
        replace(
            state,
            slots=(
                observe_packing_slot(_at(state.slots, 0), event.slot == 0, accepted, event),
                observe_packing_slot(_at(state.slots, 1), event.slot == 1, accepted, event),
                observe_packing_slot(_at(state.slots, 2), event.slot == 2, accepted, event),
                observe_packing_slot(_at(state.slots, 3), event.slot == 3, accepted, event),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_pack_failed(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.operation == state.operation
        and (
            (
                state.pending == -1
                or (
                    (
                        same_incarnation(state, event)
                        and same_session(state, event)
                        and event.slot >= 0
                        and event.slot < 4
                        and event.slot == state.pending
                        and event.operation == state.operation
                        and event.source == state.pending_source
                    )
                )
            )
        )
    )
    return (
        replace(
            state,
            slots=(
                pack_failed_slot(_at(state.slots, 0), event.slot == 0, accepted),
                pack_failed_slot(_at(state.slots, 1), event.slot == 1, accepted),
                pack_failed_slot(_at(state.slots, 2), event.slot == 2, accepted),
                pack_failed_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_packed(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.operation == state.operation
        and (
            (
                state.pending == -1
                or (
                    (
                        same_incarnation(state, event)
                        and same_session(state, event)
                        and event.slot >= 0
                        and event.slot < 4
                        and event.slot == state.pending
                        and event.operation == state.operation
                        and event.source == state.pending_source
                    )
                )
            )
        )
        and event.validated
        and event.clean
        and event.reader_excluded
        and (
            (
                (
                    (
                        same_incarnation(state, event)
                        and same_session(state, event)
                        and event.slot >= 0
                        and event.slot < 4
                        and event.slot == state.pending
                        and event.operation == state.operation
                        and event.source == state.pending_source
                        and state.write_phase == 2
                    )
                )
                or (
                    state.pending == -1 and slot_readable(state, event.slot)
                )
            )
        )
    )
    return (
        replace(
            state,
            slots=(
                packed_slot(_at(state.slots, 0), event.slot == 0, accepted),
                packed_slot(_at(state.slots, 1), event.slot == 1, accepted),
                packed_slot(_at(state.slots, 2), event.slot == 2, accepted),
                packed_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_repair(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot != state.selected
        and event.slot != state.pending
        and state.graph_phase == 0
        and slot_layout(state, event.slot)
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(
            state,
            slots=(
                repair_slot(_at(state.slots, 0), event.slot == 0, accepted),
                repair_slot(_at(state.slots, 1), event.slot == 1, accepted),
                repair_slot(_at(state.slots, 2), event.slot == 2, accepted),
                repair_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_reserve(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and state.pending == -1
        and state.graph_phase == 0
        and event.slot != state.selected
        and event.slot != state.protected
        and event.reader_excluded
        and slot_layout(state, event.slot)
    )
    return (
        replace(
            state,
            operation=state.operation + 1 if accepted else state.operation,
            pending=event.slot if accepted else state.pending,
            pending_source=event.source if accepted else state.pending_source,
            slots=(
                reserve_slot(_at(state.slots, 0), event.slot == 0, accepted, state, event),
                reserve_slot(_at(state.slots, 1), event.slot == 1, accepted, state, event),
                reserve_slot(_at(state.slots, 2), event.slot == 2, accepted, state, event),
                reserve_slot(_at(state.slots, 3), event.slot == 3, accepted, state, event),
            ),
            write_phase=1 if accepted else state.write_phase,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_rows(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
        and state.graph_phase == 2
        and event.validated
        and event.reader_excluded
    )
    return (
        replace(state, graph_phase=3 if accepted else state.graph_phase),
        OutputEffects(accepted=accepted),
    )


def apply_select(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= -1
        and event.slot < 4
        and event.slot == state.graph_slot
        and state.graph_phase != 0
        and event.operation == state.graph_operation
        and event.source == state.graph_source
        and event.output_key == state.graph_output
        and state.graph_phase == 3
        and event.validated
        and event.reader_excluded
        and complete(state, event.slot, event.source)
        and packing_clean(state, event.slot)
    )
    return (
        replace(
            state,
            graph_debt=False if accepted else state.graph_debt,
            graph_phase=0 if accepted else state.graph_phase,
            graph_slot=-1 if accepted else state.graph_slot,
            graph_valid=True if accepted else state.graph_valid,
            output_key=event.output_key if accepted else state.output_key,
            output_source=event.source if accepted else state.output_source,
            pending=(
                (
                    -1 if state.pending == event.slot else state.pending
                )
                if accepted
                else state.pending
            ),
            selected=event.slot if accepted else state.selected,
            write_phase=(
                (
                    0 if state.pending == event.slot else state.write_phase
                )
                if accepted
                else state.write_phase
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def apply_transferred(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot == state.pending
        and event.operation == state.operation
        and event.source == state.pending_source
        and state.write_phase == 2
        and event.validated
        and event.reader_excluded
        and packing_clean(state, event.slot)
    )
    return (
        replace(
            state,
            slots=(
                transferred_slot(_at(state.slots, 0), event.slot == 0, accepted),
                transferred_slot(_at(state.slots, 1), event.slot == 1, accepted),
                transferred_slot(_at(state.slots, 2), event.slot == 2, accepted),
                transferred_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
            write_phase=3 if accepted else state.write_phase,
        ),
        OutputEffects(accepted=accepted),
    )


def apply_uncertify(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (
        same_incarnation(state, event)
        and same_session(state, event)
        and event.slot >= 0
        and event.slot < 4
        and event.slot != state.pending
        and state.graph_phase == 0
    )
    return (
        replace(
            state,
            graph_valid=(
                (
                    False if event.slot == state.selected else state.graph_valid
                )
                if accepted
                else state.graph_valid
            ),
            selected=(
                (
                    -1 if event.slot == state.selected else state.selected
                )
                if accepted
                else state.selected
            ),
            slots=(
                uncertify_slot(_at(state.slots, 0), event.slot == 0, accepted),
                uncertify_slot(_at(state.slots, 1), event.slot == 1, accepted),
                uncertify_slot(_at(state.slots, 2), event.slot == 2, accepted),
                uncertify_slot(_at(state.slots, 3), event.slot == 3, accepted),
            ),
        ),
        OutputEffects(accepted=accepted),
    )


def step(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    if event.kind == 'adopt_native':
        return apply_adopt_native(state, event)
    if event.kind == 'intent':
        return apply_intent(state, event)
    if event.kind == 'protect':
        return apply_protect(state, event)
    if state.adopted and event.kind != 'invalidate':
        return state, OutputEffects()
    if event.kind == 'adopt':
        return apply_adopt(state, event)
    if event.kind == 'armed':
        return apply_armed(state, event)
    if event.kind == 'certify':
        return apply_certify(state, event)
    if event.kind == 'fail':
        return apply_fail(state, event)
    if event.kind == 'fallback':
        return apply_fallback(state, event)
    if event.kind == 'fallback_done':
        return apply_fallback_done(state, event)
    if event.kind == 'graph_begin':
        return apply_graph_begin(state, event)
    if event.kind == 'graph_fail':
        return apply_graph_fail(state, event)
    if event.kind == 'graph_repair':
        return apply_graph_repair(state, event)
    if event.kind == 'inspect':
        return apply_inspect(state, event)
    if event.kind == 'invalidate':
        return apply_invalidate(state, event)
    if event.kind == 'observe_packing':
        return apply_observe_packing(state, event)
    if event.kind == 'pack_failed':
        return apply_pack_failed(state, event)
    if event.kind == 'packed':
        return apply_packed(state, event)
    if event.kind == 'repair':
        return apply_repair(state, event)
    if event.kind == 'reserve':
        return apply_reserve(state, event)
    if event.kind == 'rows':
        return apply_rows(state, event)
    if event.kind == 'select':
        return apply_select(state, event)
    if event.kind == 'transferred':
        return apply_transferred(state, event)
    if event.kind == 'uncertify':
        return apply_uncertify(state, event)
    raise ValueError('Unknown machine event')


def apply_adopt_native(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (same_incarnation(state, event) and same_session(state, event)
                and state.pending == -1 and state.graph_phase == 0
                and event.slot >= -1 and event.slot < 4)
    return (replace(state, adopted=True, intent=event.intent, protected=event.slot,
                    selected=-1, graph_valid=False, graph_previous_valid=False,
                    graph_debt=False, output_key=0, output_source=0,
                    slots=(Slot(readability=3), Slot(readability=3),
                           Slot(readability=3), Slot(readability=3)))
            if accepted else state, OutputEffects(accepted=accepted))


def apply_intent(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (same_incarnation(state, event) and same_session(state, event)
                and (not state.adopted or event.explicit or event.intent != state.intent))
    return (replace(state, adopted=False, intent=event.intent) if accepted else state,
            OutputEffects(accepted=accepted))


def apply_protect(state: State, event: OutputEvent) -> tuple[State, OutputEffects]:
    accepted = (same_incarnation(state, event) and same_session(state, event)
                and not state.adopted and state.pending == -1 and state.graph_phase == 0
                and event.slot >= -1 and event.slot < 4 and event.reader_excluded)
    return (replace(state, protected=event.slot) if accepted else state,
            OutputEffects(accepted=accepted))


def _at(values: tuple[Slot, Slot, Slot, Slot], index: int) -> Slot:
    if not 0 <= index < len(values):
        raise ValueError('Invalid tuple index')
    return values[index]
