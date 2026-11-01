"""Foreground access to numerical contribution inspector analysis."""

from ..integration.components import Component

def create(_application):
    from ...perceptual_grade_core.model.scope_color import analysis_weights, vectorscope, output_waveform, input_analysis, selection, evaluate, evaluate_all
    scopes = _application.component('scopes.sampling')


    def get(node,space):
        view=(node.pg_chromatic.active_tone,node.pg_chromatic.family=='WARP')
        base=scopes.get(node,space,view,ui=True)
        return None if base is None else base['_color']

    def _set_export_get(value):
        nonlocal get
        get = value

    return Component(readers={
        "__name__": lambda: __name__,
        'analysis_weights': lambda: analysis_weights,
        'evaluate': lambda: evaluate,
        'evaluate_all': lambda: evaluate_all,
        'get': lambda: get,
        'input_analysis': lambda: input_analysis,
        'output_waveform': lambda: output_waveform,
        'scopes': lambda: scopes,
        'selection': lambda: selection,
        'vectorscope': lambda: vectorscope,
    }, writers={
    'get': _set_export_get})
