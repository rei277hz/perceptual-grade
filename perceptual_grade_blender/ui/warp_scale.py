"""Paired Scale/Chroma editing; grading and exact saved-value writes stay pure.

The caller freezes ``model`` at gesture start.  Increasing Scale may reduce
Chroma Stops to stay on the increasing-lightness part of a negative-energy
ray. Decreasing Scale never compensates chroma merely to retain display radius.
"""

from ..integration.components import Component

def create(_application):
    import math
    import numpy as np
    from ...perceptual_grade_core.model import color
    controls = _application.component('model.controls')
    from ...color_core.modcam16 import signed_hk
    _candidate = _application.component('ui.warp_drag')._candidate

    TURNING_RATIO = 33.*math.sqrt(3.)
    _SAFETY = 2.e-6  # Remain before the turning point after RNA float32 storage.


    def _height_floor(raw, factor):
        """Maximum height reached along the unchanged upward Scale ray."""
        c = float(np.hypot(*raw.q[1:])); e = float(raw.energy)
        if e < 0:
            factor = max(1., min(factor, TURNING_RATIO*c/abs(e)))
        return max(0., float(signed_hk(np.sign(e)*math.sqrt(abs(e))*factor, c*factor)))


    def edit(model, tone, handle, requested_scale, start_params=None):
        """Return one validated Scale/Chroma pair, or ``{}`` for a rejected edit.

        ``start_params`` optionally supplies the frozen gesture snapshot when the
        caller already has a model for the current live settings.  Returned values
        retain ideal precision, but acceptance also checks their stored float32
        representation. Hue is never changed by this transaction.
        """
        if start_params is not None:
            model = color.Model(model.tone, start_params)
        if not 0 <= tone < 4 or not 0 <= handle < 6:
            return {}
        scale_name = f'web_scale_{tone}_{handle}'
        chroma_name = f'web_chroma_{tone}_{handle}'
        start_scale = float(getattr(model.params, scale_name))
        start_chroma = float(getattr(model.params, chroma_name))
        requested_scale = float(requested_scale)
        lo, hi = controls.meta(scale_name)[3:5]
        if not math.isfinite(requested_scale) or not lo <= requested_scale <= hi:
            return {}
        # Reversing a gesture back to its origin must restore its exact pair, even
        # when the original saved handle is outside the positive-height domain.
        if requested_scale == start_scale:
            return {scale_name: start_scale, chroma_name: start_chroma}
        upward = requested_scale > start_scale
        try:
            evaluator = model.guarded_corner_evaluator(tone, handle)
            original_raw, _ = evaluator()
            minimum_h = _height_floor(original_raw, math.exp2(requested_scale-start_scale)) if upward else 0.
            evaluator.scale = math.exp2(requested_scale)

            def acceptable(raw, point, height_floor=minimum_h):
                e = float(raw.energy); c = float(np.hypot(*raw.q[1:]))
                if (not np.isfinite(raw.q).all() or not math.isfinite(e) or
                        raw.q[0] <= 0 or not np.isfinite(point).all()):
                    return False
                if not upward:
                    return True
                before_peak = e >= 0 or -e <= TURNING_RATIO*c*(1.-_SAFETY)
                return before_peak and float(raw.q[0]) >= height_floor-1.e-9*max(1., height_floor)

            def feasible(gain):
                try:
                    return acceptable(*evaluator(gain))
                except (ValueError, ArithmeticError):
                    return False

            gain = start_chroma
            if not feasible(gain):
                if not upward:
                    return {}
                # Adaptance need not make this residual globally monotone. Search
                # downward in deterministic local steps, refining the first safe
                # bracket instead of assuming the entire control range is one ray.
                minimum = controls.meta(chroma_name)[3]
                upper = gain
                for step in range(1, int(math.ceil((gain-minimum)/.125))+1):
                    lower = max(minimum, gain-.125*step)
                    if feasible(lower):
                        break
                    upper = lower
                else:
                    return {}
                for _ in range(36):
                    middle = (lower+upper)/2
                    if feasible(middle):lower = middle
                    else:upper = middle
                gain = lower

            values = {scale_name: requested_scale, chroma_name: gain}
            # Complete-model validation catches global hue ordering; rounded
            # validation prevents an accepted ideal edit becoming invalid in RNA.
            for stored in (False, True):
                candidate_values = {name: float(np.float32(value)) if stored else value
                                    for name, value in values.items()}
                candidate = _candidate(model, candidate_values)
                raw = candidate.warp_targets(tone)
                point = candidate.guarded_saturation_targets(tone)[handle+1]
                selected = type(raw)(raw.q[handle+1], raw.energy[handle+1])
                # Compare stored values to the same ray at the stored Scale; RNA
                # rounding of the independent slider is not a loss of lightness.
                stored_floor = _height_floor(original_raw, math.exp2(candidate_values[scale_name]-start_scale)) if upward else 0.
                if not acceptable(selected, point, stored_floor):
                    # A final float32 rounding can consume the lightness margin.
                    # Probe one representable chroma step down, retaining the
                    # exact requested Scale and the same acceptance conditions.
                    if not stored or not upward:
                        return {}
                    saved_gain = np.float32(candidate_values[chroma_name])
                    saved_gain = np.nextafter(saved_gain, np.float32(-np.inf))
                    values[chroma_name] = float(saved_gain)
                    candidate_values[chroma_name] = float(saved_gain)
                    candidate = _candidate(model, candidate_values)
                    raw = candidate.warp_targets(tone)
                    selected = type(raw)(raw.q[handle+1], raw.energy[handle+1])
                    if not acceptable(selected, candidate.guarded_saturation_targets(tone)[handle+1], stored_floor):
                        return {}
            return values
        except (ValueError, ArithmeticError):
            return {}

    def _set_export_TURNING_RATIO(value):
        nonlocal TURNING_RATIO
        TURNING_RATIO = value

    def _set_export__SAFETY(value):
        nonlocal _SAFETY
        _SAFETY = value

    def _set_export__height_floor(value):
        nonlocal _height_floor
        _height_floor = value

    def _set_export_edit(value):
        nonlocal edit
        edit = value

    return Component(readers={
        "__name__": lambda: __name__,
        'TURNING_RATIO': lambda: TURNING_RATIO,
        '_SAFETY': lambda: _SAFETY,
        '_candidate': lambda: _candidate,
        '_height_floor': lambda: _height_floor,
        'color': lambda: color,
        'controls': lambda: controls,
        'edit': lambda: edit,
        'math': lambda: math,
        'np': lambda: np,
        'signed_hk': lambda: signed_hk,
    }, writers={
        'TURNING_RATIO': _set_export_TURNING_RATIO,
        '_SAFETY': _set_export__SAFETY,
    '_height_floor': _set_export__height_floor, 'edit': _set_export_edit})
