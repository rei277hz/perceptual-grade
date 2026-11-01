"""Blender control bounds/formatting, separate from cached grading math.

The shader accepts live, unconstrained coefficients. These specifications are
shared by RNA registration and diagram interaction, so their limits agree.
"""

from ..integration.components import Component

def create(_application):
    from ...perceptual_grade_core.localization.controls import control_tooltip, control_label
    M = _application.component('localization.i18n').M
    import math
    from ...perceptual_grade_core.model import color as ch

    CONTROLS = tuple((name, M(control_label(name)), *spec[2:])
                     for spec in ch.CONTROLS for name in (spec[0],))
    META = {spec[0]: spec for spec in CONTROLS}

    OVERRIDES={}
    for family,limit in (('j',50.),('c',4.)):
        for axis in 'ab':
            for tone in range(4):
                field=f'{family}_{axis}_{tone}';old=META[field]
                OVERRIDES[field]=(*old[:3],-limit,limit,*old[5:])


    def meta(field):return OVERRIDES.get(field,META[field])


    def step(field):
        if field=='adaptance_half_saturation':return .05
        return 1. if meta(field)[5]==0 else None


    def snap(field,value):
        spec=meta(field);lo,hi=spec[3:5];value=max(lo,min(hi,value));size=step(field)
        if size is not None:
            value=lo+math.floor((value-lo)/size+.5+1e-9)*size
            value=round(max(lo,min(hi,value)),spec[5])
        return int(value) if spec[5]==0 else value


    def tooltip_id(field):
        return control_tooltip('color', field)

    def _set_export_CONTROLS(value):
        nonlocal CONTROLS
        CONTROLS = value

    def _set_export_META(value):
        nonlocal META
        META = value

    def _set_export_OVERRIDES(value):
        nonlocal OVERRIDES
        OVERRIDES = value

    def _set_export_field(value):
        nonlocal field
        field = value

    def _set_export_old(value):
        nonlocal old
        old = value

    def _set_export_meta(value):
        nonlocal meta
        meta = value

    def _set_export_step(value):
        nonlocal step
        step = value

    def _set_export_snap(value):
        nonlocal snap
        snap = value

    def _set_export_tooltip_id(value):
        nonlocal tooltip_id
        tooltip_id = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CONTROLS': lambda: CONTROLS,
        'M': lambda: M,
        'META': lambda: META,
        'OVERRIDES': lambda: OVERRIDES,
        'axis': lambda: axis,
        'ch': lambda: ch,
        'control_label': lambda: control_label,
        'control_tooltip': lambda: control_tooltip,
        'family': lambda: family,
        'field': lambda: field,
        'limit': lambda: limit,
        'math': lambda: math,
        'meta': lambda: meta,
        'old': lambda: old,
        'snap': lambda: snap,
        'spec': lambda: spec,
        'step': lambda: step,
        'tone': lambda: tone,
        'tooltip_id': lambda: tooltip_id,
    }, writers={
        'CONTROLS': _set_export_CONTROLS,
        'META': _set_export_META,
        'OVERRIDES': _set_export_OVERRIDES,
        'field': _set_export_field,
        'old': _set_export_old,
    'meta': _set_export_meta, 'step': _set_export_step, 'snap': _set_export_snap, 'tooltip_id': _set_export_tooltip_id})
