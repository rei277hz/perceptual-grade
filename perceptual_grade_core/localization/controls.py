"""Message IDs associated with mathematical control identifiers."""
from pathlib import Path
import json

_BINDINGS = json.loads((Path(__file__).parent/'bindings.json').read_text(encoding='utf-8'))


def control_tooltip(stage, field):
    return _BINDINGS[f'{stage}.{field}']


_PRESENTATION = json.loads((Path(__file__).parent/'control-labels.json').read_text(encoding='utf-8'))


def control_label(field):
    """Resolve a control field to its current semantic label ID."""
    return _PRESENTATION['labels'][field]


def control_readout(stem):
    """Resolve a coefficient stem to its semantic diagram-readout ID."""
    return _PRESENTATION['readouts'][stem]
