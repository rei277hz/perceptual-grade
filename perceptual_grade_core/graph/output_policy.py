"""Output-authority vocabulary and pure observation queries.

Native adapters advance the output machine through their App Store.
"""
from .policies import output as _output

SCHEMA = 3
SLOT_COUNT = 4
UNKNOWN, READABLE, UNREADABLE, EXTERNAL = range(4)
IDLE, RESERVED, ARMED, READY = range(4)
WRITE_IDLE, WRITE_RESERVED, WRITE_ARMED, WRITE_READY = IDLE, RESERVED, ARMED, READY
GRAPH_IDLE, PREPARED, FALLBACK, ROWS, GRAPH_REPAIR = range(5)
PACK_UNKNOWN, PACK_CLEAN, PACK_PENDING = range(3)
KINDS = ('reserve', 'armed', 'transferred', 'fail', 'inspect', 'certify', 'repair',
         'graph_begin', 'fallback', 'rows', 'select', 'graph_fail', 'graph_repair',
         'adopt', 'uncertify', 'fallback_done', 'invalidate',
         'observe_packing', 'packed', 'pack_failed', 'adopt_native', 'intent', 'protect')

Slot, State, Event, Effects = _output.Slot, _output.State, _output.OutputEvent, _output.OutputEffects
slot_for = _output.slot_for
complete = _output.complete
output_complete = _output.output_complete
packing_clean = _output.packing_clean


def establish(incarnation, uncertain=False, *, session=0):
    return _output.establish(incarnation, uncertain, session)
