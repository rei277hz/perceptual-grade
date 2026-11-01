"""Explicit numerical domain records carried by the execution transport.

Models cross the boundary as settings, never locks, kernels or Python pickles.
Scope envelopes use a flat table of whitelisted records because inspector views
share arrays and contain backreferences to their base and selection dictionaries.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
import numpy as np

VERSION = 1


def freeze_space(space):
    from ..model.linear import GradeSpace
    if isinstance(space, GradeSpace):
        return {'kind': 'grade', 'preset': space.preset}
    if isinstance(space, str):
        return {'kind': 'named', 'name': space}
    raise TypeError('Unsupported grading space')


def thaw_space(value):
    from ..model.linear import GradeSpace
    if value['kind'] == 'grade':
        return GradeSpace(value['preset'])
    if value['kind'] == 'named':
        return value['name']
    raise ValueError('Unknown grading space record')


def _parameters(value):
    return {field.name: getattr(value, field.name) for field in fields(value)}


def freeze_model(model):
    from ..model.transforms import Stack
    if not isinstance(model, Stack):
        raise TypeError('LUT execution requires a numerical Stack')
    return {'version': VERSION, 'space': freeze_space(model.space),
            'directions': model.directions,
            'entries': tuple({'matrix': tuple(tuple(row) for row in entry.matrix),
                              'tone': _parameters(entry.tone),
                              'color': _parameters(entry.color),
                              'space': freeze_space(entry.space)}
                             for entry in model.entries)}


def thaw_model(value):
    from ..model import tone, color
    from ..model.transforms import Settings, Stack
    _version(value)
    entries = tuple(Settings(tuple(tuple(row) for row in entry['matrix']),
                             tone.Parameters(**entry['tone']), color.Parameters(**entry['color']),
                             thaw_space(entry['space'])) for entry in value['entries'])
    return Stack(entries, thaw_space(value['space']), tuple(value['directions']))


def freeze_cube(cube):
    result = {'version': VERSION, 'values': cube.values, 'signature': cube.signature,
              'space': freeze_space(cube.space), 'size': cube.size,
              'diagnostics': dict(cube.diagnostics)}
    if getattr(cube, 'prepared_pixels', None) is not None:
        result['prepared_pixels'] = cube.prepared_pixels
    return result


def thaw_cube(value):
    from ..model.lut import Cube
    _version(value)
    result = Cube(value['values'], value['signature'], thaw_space(value['space']), value['size'])
    result.diagnostics = dict(value['diagnostics'])
    if 'prepared_pixels' in value:
        result.prepared_pixels = value['prepared_pixels']
    return result


def _version(value):
    if value.get('version') != VERSION:
        raise ValueError('Unsupported numerical payload version')


def _freeze_scope(value):
    from ..model import color, tone, linear, lut, transforms
    from ...color_core import modcam16 as responses
    from ..scopes.identity import ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity, validate_identity
    records, memo = [], {}
    parameter_types = {tone.Parameters: 'tone_parameters', color.Parameters: 'color_parameters',
                       linear.Parameters: 'linear_parameters'}
    identity_types = {ScopeIdentity: 'scope_identity',
                      ActiveScopeIdentity: 'active_scope_identity',
                      LinearScopeIdentity: 'linear_scope_identity'}

    def encode(item):
        if item is None or isinstance(item, (str, bool, int, float)):
            return {'value': item}
        if isinstance(item, np.generic):
            return {'value': item.item()}
        if id(item) in memo:
            return {'ref': memo[id(item)]}
        index = len(records)
        memo[id(item)] = index
        records.append(None)
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError('Scope payload arrays must have a numerical dtype')
            kind, data = 'array', item
        elif isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError('Scope record keys must be strings')
            kind, data = 'mapping', {key: encode(entry) for key, entry in item.items()}
        elif isinstance(item, (tuple, list)):
            kind, data = 'tuple', tuple(encode(entry) for entry in item)
        elif type(item) in parameter_types:
            kind, data = parameter_types[type(item)], _parameters(item)
        elif type(item) in identity_types:
            validate_identity(item)
            kind = identity_types[type(item)]
            data = {field.name: encode(getattr(item, field.name))
                    for field in fields(item)}
        elif isinstance(item, linear.GradeSpace):
            kind, data = 'space', freeze_space(item)
        elif isinstance(item, tone.Model):
            kind, data = 'tone_model', {'params': encode(item.params), 'space': encode(item.space)}
        elif isinstance(item, color.Model):
            kind, data = 'color_model', {'tone': encode(item.tone), 'params': encode(item.params)}
        elif isinstance(item, transforms.Grade):
            settings=item.settings
            kind, data = 'grade_model', {
                'matrix': tuple(tuple(row) for row in settings.matrix),
                'tone': encode(settings.tone), 'color': encode(settings.color),
                'space': encode(settings.space)}
        elif isinstance(item, responses.Appearance):
            kind, data = 'appearance', {'q': encode(item.q), 'ja': encode(item.ja)}
        elif isinstance(item, responses.State):
            kind, data = 'state', {'q': encode(item.q), 'energy': encode(item.energy)}
        elif isinstance(item, lut.Cube):
            kind, data = 'cube', freeze_cube(item)
        else:
            raise TypeError('Unsupported scope payload object: ' + type(item).__name__)
        records[index] = {'kind': kind, 'data': data}
        return {'ref': index}

    root = encode(value)
    return {'version': VERSION, 'root': root, 'records': records}


def _thaw_scope(value):
    from ..model import color, tone, linear, transforms
    from ...color_core import modcam16 as responses
    from ..scopes.identity import ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity, validate_identity
    _version(value)
    records, memo, active = value['records'], {}, set()
    parameter_types = {'tone_parameters': tone.Parameters, 'color_parameters': color.Parameters,
                       'linear_parameters': linear.Parameters}
    identity_types = {'scope_identity': ScopeIdentity,
                      'active_scope_identity': ActiveScopeIdentity,
                      'linear_scope_identity': LinearScopeIdentity}

    def decode(token):
        if 'value' in token:
            return token['value']
        index = token['ref']
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError('Invalid scope record reference')
        if index in memo:
            return memo[index]
        if index in active:
            raise ValueError('Unsupported cycle in a scope record')
        active.add(index)
        record = records[index]
        kind, data = record['kind'], record['data']
        if kind == 'mapping':
            result = memo[index] = {}
            result.update((key, decode(entry)) for key, entry in data.items())
        elif kind == 'tuple':
            result = tuple(decode(entry) for entry in data)
        elif kind == 'array':
            if not isinstance(data, np.ndarray) or data.dtype.hasobject:
                raise ValueError('Invalid scope array')
            result = data
        elif kind in parameter_types:
            result = parameter_types[kind](**data)
        elif kind in identity_types:
            if set(data) != {field.name for field in fields(identity_types[kind])}:
                raise ValueError('Invalid scope identity fields')
            result = identity_types[kind](**{
                field.name: decode(data[field.name])
                for field in fields(identity_types[kind])})
            validate_identity(result)
        elif kind == 'space':
            result = thaw_space(data)
        elif kind == 'tone_model':
            result = tone.model(decode(data['params']), decode(data['space']))
        elif kind == 'color_model':
            result = color.Model(decode(data['tone']), decode(data['params']))
        elif kind == 'grade_model':
            settings=transforms.Settings(tuple(tuple(row) for row in data['matrix']),
                decode(data['tone']), decode(data['color']), decode(data['space']))
            result=transforms.Grade(settings)
        elif kind == 'appearance':
            result = responses.Appearance(decode(data['q']), decode(data['ja']))
        elif kind == 'state':
            result = responses.State(decode(data['q']), decode(data['energy']))
        elif kind == 'cube':
            result = thaw_cube(data)
        else:
            raise ValueError('Unknown scope record kind: ' + str(kind))
        active.remove(index)
        memo[index] = result
        return result

    return decode(value['root'])


def readonly(value, memo=None):
    """Freeze arrays and inspector mappings while retaining their aliases."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, Mapping):
        result = {}
        memo[id(value)] = proxy = MappingProxyType(result)
        result.update((key, readonly(item, memo)) for key, item in value.items())
        return proxy
    elif isinstance(value, (tuple, list)):
        result = tuple(readonly(item, memo) for item in value)
        memo[id(value)] = result
        return result
    elif is_dataclass(value):
        for field in fields(value):
            readonly(getattr(value, field.name), memo)
    return value


def freeze_scope_request(snapshot):
    return _freeze_scope(snapshot)


def thaw_scope_request(payload):
    return _thaw_scope(payload)


def freeze_scope_result(result):
    return _freeze_scope(result)


def freeze_full_scope_result(result):
    """Transport display data, retaining the rich analysis graph in the helper.

    Project every mapping that can refer back to numerical intermediates. In
    particular, copying a selection's original _base or _input would silently
    send the entire analysis graph again. Arrays stay shared until the existing
    transport/ownership boundary copies them; projection never mutates caches.
    """
    def take(value, names):
        return {name: value[name] for name in names if name in value}

    def distribution(value):
        return take(value, ('density', 'histogram', 'invalid', 'above', 'below', 'count'))

    def vectorscope(value):
        return take(value, ('density', 'outside'))

    texture_only = '_warp_clouds' in result
    base = take(result, ('model', 'label', 'signature', '_scope_key', '_scope_provenance',
                         '_scope_evaluation', '_warp_clouds', '_warp_cpu_seconds', '_warp_cpu_reused'))
    base.update(before=distribution(result['before']), after=distribution(result['after']),
                linear_vectorscope=vectorscope(result['linear_vectorscope']))
    # Tone-only requests stop after Stage 2.  They intentionally carry no
    # chromatic selections, so keep the compact base and optional comparison
    # signal without touching the Stage 3 graph.
    if 'stage' in result and result['stage'] == 2:
        base['stage'] = 2
        comparison = result.get('_comparison')
        if comparison is None:
            base['_comparison'] = None
        else:
            signal = take(comparison, ('bounded_saturation', 'energy'))
            signal['distribution'] = distribution(comparison['distribution'])
            signal['vectorscope'] = vectorscope(comparison['vectorscope'])
            base['_comparison'] = signal
        base['_comparison_selected'] = None
        return _freeze_scope(base)
    selections = {}
    for index, source in enumerate(result['_colors']):
        names = ('model', 'sectors', 'waveform_selected', 'waveform_outside', 'label', 'saturation_excluded')
        if not texture_only:
            names += ('weights', 'after_energy', 'bounded_saturation')
            if source.get('sample_mass') is not None and np.any(source['sample_mass'] != 1):
                names += ('opacity_weights', 'sample_mass')
        selected = take(source, names)
        if texture_only:
            selected['_warp_clouds'] = base['_warp_clouds']
            selected['_warp_tone'] = index
        selected['histograms'] = take(source['histograms'], ('low', 'high', 'a', 'b', 'tone'))
        selected['vectorscope'] = vectorscope(source['vectorscope'])
        neutral = source['neutral']
        selected['neutral'] = None if neutral is None else take(neutral, (
            'center', 'covariance', 'ellipse_covariance', 'count', 'reference'))
        selected['_base'] = base
        selections[id(source)] = selected
    base['_colors'] = tuple(selections[id(source)] for source in result['_colors'])
    base['_color'] = selections[id(result['_color'])]

    comparison = result['_comparison']
    signal = None
    if comparison is not None:
        signal = {} if texture_only else take(comparison, ('bounded_saturation', 'energy'))
        signal['distribution'] = distribution(comparison['distribution'])
        signal['vectorscope'] = vectorscope(comparison['vectorscope'])
    base['_comparison'] = signal
    base['_comparison_selected'] = None
    # A failed reference selection can leave a valid common signal but no
    # per-tone selections. Preserve that optional shape for foreground views.
    if '_comparisons_selected' in result:
        comparisons = {}
        for source in result['_comparisons_selected']:
            names = ('waveform',) if texture_only else ('weights', 'waveform')
            if not texture_only and source.get('sample_mass') is not None and np.any(source['sample_mass'] != 1):
                names += ('opacity_weights', 'sample_mass')
            selected = take(source, names)
            selected.update(signal=signal, vectorscope=vectorscope(source['vectorscope']))
            comparisons[id(source)] = selected
        base['_comparisons_selected'] = tuple(comparisons[id(source)]
                                              for source in result['_comparisons_selected'])
        if result['_comparison_selected'] is not None:
            base['_comparison_selected'] = comparisons[id(result['_comparison_selected'])]
    return _freeze_scope(base)


def thaw_scope_result(payload):
    return readonly(_thaw_scope(payload))
