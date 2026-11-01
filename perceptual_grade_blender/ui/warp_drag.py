"""Transactional Warp gestures, independent of Blender events and publication.

Position mode follows the guarded compact chart. At raw carried E=0 the
remainder of the pointer segment changes log2 pre-Adaptance chroma instead.
The virtual cursor never snaps to the guarded marker in that mode. Only a
successful complete edit updates the caller's transient state.
"""

from ..integration.components import Component

def create(_application):
    from dataclasses import dataclass, replace
    import math
    import numpy as np
    from ...perceptual_grade_core.model import color
    controls = _application.component('model.controls')

    PIXELS_PER_STOP = 40.
    MIN_CURSOR_PIXELS = 16.


    def _candidate(model, values):
        if any(not np.isfinite(v) or not controls.meta(n)[3] <= v <= controls.meta(n)[4]
               for n, v in values.items()):
            raise ValueError('Warp target exceeds its control range')
        candidate = color.Model(model.tone, replace(model.params, **values))
        angles = candidate.web.parameters(candidate.web.geometry.anchors)['target_angles']
        gaps = np.diff(np.concatenate((angles, angles[..., :1]+2*np.pi), axis=-1), axis=-1)
        if np.min(gaps) < 1.e-4:
            raise ValueError('Warp target would swap adjacent hues')
        return candidate


    def _point(model, row, handle):
        return model.guarded_saturation_targets(row)[handle+1].copy()


    def _energy(model, row, handle):
        state = model.warp_targets(row)
        return _relative_energy(state.q[handle+1], state.energy[handle+1])


    def _relative_energy(q, energy):
        e = float(energy)
        if not np.isfinite(q).all() or not math.isfinite(e) or q[0] <= 0:
            raise ValueError('Warp drag requires a finite positive raw height')
        return e/max(1., q[0]*q[0], 66.*np.hypot(*q[1:]))


    @dataclass(frozen=True)
    class _Probe:
        settings: object
        raw: object
        point: object


    def _probe(evaluator, settings):
        """Evaluate just the selected source corner; defer full-model validation."""
        row, handle = evaluator.row, evaluator.handle
        angle = evaluator.web.geometry.angles[row, handle]+math.radians(settings.hue_degrees[row][handle])
        raw, point = evaluator(settings.log_chroma[row][handle], angle)
        return _Probe(settings, raw, point)


    def _accept(model, probe, row, handle):
        values = {f'web_{family}_{row}_{handle}': getattr(probe.settings, field)[row][handle]
                  for family, field in [('chroma', 'log_chroma'), ('hue', 'hue_degrees'), ('scale', 'log_scale')]}
        return _candidate(model, values)


    def _crossing(function, low_value, low_probe, high_value, high_probe, tolerance, power=1., positive=True):
        """Bracketed secant with periodic bisection and a positive-side result.

        Probe tolerances are display/normalized-energy precision, not an arbitrary
        fixed number of expensive complete-model reevaluations.
        """
        if low_value >= 0:
            return 0., low_probe
        if high_value < 0:
            raise ValueError('No physical-boundary crossing in this pointer segment')
        low, high = 0., 1.
        for index in range(64):
            width = high-low
            if not positive and abs(low_value) <= tolerance:
                return low, low_probe
            # Near a cone-response fold, binary64 residuals can stop changing
            # before the requested energy tolerance. The bracket itself is the
            # reliable quantity; retain the requested side and continue the rest
            # of the gesture from that side of E=0.
            if high_value <= tolerance or width < 2.e-10:
                return high, high_probe
            # Active Adaptance carries E=sign(A)|A|^(2 gamma). Interpolate its
            # signed response rather than an artificially flat energy residual.
            left, right = abs(low_value)**(1/power), high_value**(1/power)
            fraction = low+left*width/(left+right)
            # Interpolation is fast for a smooth ray; forced bisections retain the
            # bracket's convergence when a very shallow side stalls the secant.
            if index % 2 == 1 or not low+width*1.e-6 < fraction < high-width*1.e-6:
                fraction = (low+high)/2
            value, probe = function(fraction)
            if value >= 0:
                high, high_value, high_probe = fraction, value, probe
            else:
                low, low_value, low_probe = fraction, value, probe
        raise ValueError('Warp boundary crossing did not converge')


    def reset(model, action):
        """Rebase after R or mouse-down, without changing the frozen view."""
        row, handle = action['tone'], action['handle']
        point = _point(model, row, handle)
        action.pop('disabled', None)
        if handle >= 0:
            raw = model.warp_targets(row)
            q = raw.q[handle+1]
            if not np.isfinite(q).all() or q[0] <= 0:
                # Negative-H colors remain under the existing black policy. They
                # cannot be represented by this positive-height Warp target solve;
                # keep the gesture harmless until the user resets that handle.
                action.update(target=point, cursor=point.copy(), mode='position',
                              rim_active=False, disabled=True)
                action.pop('stops_origin', None)
                return
            mode = 'stops' if _energy(model, row, handle) <= 0 else 'position'
        else:
            mode = 'position'
        action.update(target=point, cursor=point.copy(), mode=mode, rim_active=mode == 'stops')
        if mode == 'stops':
            cursor = _stops_cursor(model, action, point)
            action['cursor'] = cursor
            action['stops_origin'] = (float(np.hypot(*cursor)), getattr(model.params, f'web_chroma_{row}_{handle}'))
        else:
            action.pop('stops_origin', None)


    def _stops_cursor(model, action, point):
        """Give a near-neutral marker a usable angular lever without moving it."""
        radius = float(np.hypot(*point))
        plot = action['plot']
        minimum = MIN_CURSOR_PIXELS*action.get('scale', 1.)*(plot.ymax-plot.ymin)/plot.rect[3]
        if radius >= minimum:
            return point.copy()
        if radius > 0:
            direction = point/radius
        else:
            row, handle = action['tone'], action['handle']
            angle = model.web.parameters(model.web.geometry.anchors[row])['target_angles'][handle]
            direction = np.array([np.cos(angle), np.sin(angle)])
        return minimum*direction


    def _position(model, row, handle, target):
        values = model.target_guarded_saturation_values(row, handle, target)
        candidate = _candidate(model, values)
        point = _point(candidate, row, handle)
        if not np.isfinite(point).all() or np.linalg.norm(point-target) > 2.e-7:
            raise ValueError('Warp position solve did not reach its target')
        return candidate


    def _boundary(evaluator, point):
        """Signed radial distance from this direction's guarded image of raw E=0."""
        angle = math.atan2(point[1], point[0])
        probe = _probe(evaluator, evaluator.boundary(angle))
        return float(np.hypot(*point)-np.hypot(*probe.point)), probe


    def _enter(model, row, handle, start, delta):
        # Solving the physical boundary directly also handles a single mouse event
        # whose destination lies outside the chart. Never unproject radius one.
        evaluator = model.guarded_corner_evaluator(row, handle)
        evaluate = lambda fraction: _boundary(evaluator, start+fraction*delta)
        lo_t = 0.
        low, beginning = evaluate(0.)
        # An exit from Stops may start exactly on the rim and travel inward before
        # reaching the rim again. Skip that departing root; find the next outward
        # crossing instead of bouncing between modes at fraction zero.
        denominator = float(np.dot(delta, delta))
        nearest = float(-np.dot(start, delta)/denominator) if denominator else 0.
        samples = [nearest] if 0. < nearest < 1. else []
        if evaluator.adaptation.active:samples.extend((.25, .5, .75))
        for hi_t in sorted(set(samples+[1.])):
            high, end = evaluate(hi_t)
            if high >= 0:
                break
            lo_t, low, beginning = hi_t, high, end
        else:
            raise ValueError('No physical-boundary crossing in this pointer segment')
        segment = lambda value:evaluate(lo_t+value*(hi_t-lo_t))
        local, probe = _crossing(segment, low, beginning, high, end, 1.e-10, positive=False)
        fraction = lo_t+local*(hi_t-lo_t)
        return fraction, _accept(model, probe, row, handle)


    def _stops(evaluator, cursor, origin, units_per_stop):
        radius = float(np.hypot(*cursor))
        gain = origin[1]+(radius-origin[0])/units_per_stop
        angle = math.atan2(cursor[1], cursor[0])
        return _probe(evaluator, evaluator.hue_target(gain, angle))


    def _stops_exit_bracket(evaluate, start, delta, initial, ending, active_adaptation):
        """Find an interior re-entry when an entire pointer chord is outside.

        With an active Adaptance the rim also varies with hue, so probe the chord
        midpoint/quarters in addition to its nearest point to the neutral axis.
        """
        denominator = float(np.dot(delta, delta))
        nearest = float(-np.dot(start, delta)/denominator) if denominator else 0.
        samples = [nearest] if 0. < nearest < 1. else []
        if active_adaptation:samples.extend((.25, .5, .75))
        prior_t, (prior_energy, prior_probe) = 0., initial
        for fraction in sorted(set(samples+[1.])):
            energy, probe = ending if fraction == 1. else evaluate(fraction)
            if prior_energy <= 1.e-10 and energy > 1.e-10:
                return prior_t, prior_energy, prior_probe, fraction, energy, probe
            prior_t, prior_energy, prior_probe = fraction, energy, probe
        return None


    def move(model, action, dx, dy):
        """Return accepted RNA values; rejected edits leave all gesture state intact."""
        if action.get('disabled') or (dx == 0 and dy == 0):
            return {}
        state = dict(action)
        row, handle = state['tone'], state['handle']
        if 'cursor' not in state:
            reset(model, state)
        p = state['plot']; speed = .2 if handle < 0 else .5
        delta = speed*np.array([dy*(p.ymax-p.ymin)/p.rect[3], -dx*(p.xmax-p.xmin)/p.rect[2]])
        units_per_stop = .5*(p.ymax-p.ymin)/p.rect[3]*PIXELS_PER_STOP*state.get('scale', 1.)
        original = model
        try:
            for _ in range(6):
                start = state['cursor']; target = start+delta
                if state['mode'] == 'position':
                    try:
                        candidate = _position(model, row, handle, target)
                    except (ValueError, ArithmeticError):
                        if handle < 0:
                            raise
                        candidate = None
                    if candidate is not None and (handle < 0 or _energy(candidate, row, handle) > 0):
                        model = candidate
                        state['cursor'] = target
                        break
                    fraction, model = _enter(model, row, handle, start, delta)
                    point = _point(model, row, handle)
                    cursor = _stops_cursor(model, state, point)
                    state.update(cursor=cursor, mode='stops', rim_active=True,
                                 stops_origin=(float(np.hypot(*cursor)), getattr(model.params, f'web_chroma_{row}_{handle}')))
                else:
                    origin = state['stops_origin']
                    evaluator = model.guarded_corner_evaluator(row, handle)
                    def evaluate(fraction):
                        probe = _stops(evaluator, start+fraction*delta, origin, units_per_stop)
                        return _relative_energy(probe.raw.q, probe.raw.energy), probe
                    energy, probe = evaluate(1.)
                    initial_energy, initial_probe = evaluate(0.)
                    bracket = _stops_exit_bracket(evaluate, start, delta,
                        (initial_energy, initial_probe), (energy, probe), evaluator.adaptation.active)
                    if bracket is None:
                        model = _accept(model, probe, row, handle)
                        state['cursor'] = target
                        break
                    # Find the exit in control space; Adaptance can make hue-only
                    # motion cross this boundary too. No radius proxy is used.
                    lo_t, lo_e, lo_probe, hi_t, hi_e, hi_probe = bracket
                    segment=lambda value:evaluate(lo_t+value*(hi_t-lo_t))
                    local, probe = _crossing(segment, lo_e, lo_probe, hi_e, hi_probe,
                                             1.e-10, evaluator.power)
                    fraction = lo_t+local*(hi_t-lo_t)
                    model = _accept(model, probe, row, handle)
                    state.update(cursor=_point(model, row, handle), mode='position', rim_active=False)
                    state.pop('stops_origin', None)
                delta = delta*(1-fraction)
                if np.linalg.norm(delta) < 1.e-11:
                    break
            else:
                raise ValueError('Warp gesture crossed too many local branches')
            names = action['fields']
            values = {name: getattr(model.params, name) for name in names}
            # RNA stores float32. Reject unsafe rounding before changing parameters
            # or the cursor; ideal control values retain sub-event accumulation.
            saved = _candidate(original, {name: float(np.float32(value)) for name, value in values.items()})
            actual = _point(saved, row, handle)
            ideal = _point(model, row, handle)
            if not np.isfinite(actual).all() or np.linalg.norm(actual-ideal) > 2.e-5:
                raise ValueError('Warp target is unstable at stored precision')
        except (ValueError, ArithmeticError):
            return {}
        state['target'] = actual
        action.update(state)
        return values

    def _set_export_MIN_CURSOR_PIXELS(value):
        nonlocal MIN_CURSOR_PIXELS
        MIN_CURSOR_PIXELS = value

    def _set_export_PIXELS_PER_STOP(value):
        nonlocal PIXELS_PER_STOP
        PIXELS_PER_STOP = value

    def _set_export__candidate(value):
        nonlocal _candidate
        _candidate = value

    def _set_export__point(value):
        nonlocal _point
        _point = value

    def _set_export__energy(value):
        nonlocal _energy
        _energy = value

    def _set_export__relative_energy(value):
        nonlocal _relative_energy
        _relative_energy = value

    def _set_export__Probe(value):
        nonlocal _Probe
        _Probe = value

    def _set_export__probe(value):
        nonlocal _probe
        _probe = value

    def _set_export__accept(value):
        nonlocal _accept
        _accept = value

    def _set_export__crossing(value):
        nonlocal _crossing
        _crossing = value

    def _set_export_reset(value):
        nonlocal reset
        reset = value

    def _set_export__stops_cursor(value):
        nonlocal _stops_cursor
        _stops_cursor = value

    def _set_export__position(value):
        nonlocal _position
        _position = value

    def _set_export__boundary(value):
        nonlocal _boundary
        _boundary = value

    def _set_export__enter(value):
        nonlocal _enter
        _enter = value

    def _set_export__stops(value):
        nonlocal _stops
        _stops = value

    def _set_export__stops_exit_bracket(value):
        nonlocal _stops_exit_bracket
        _stops_exit_bracket = value

    def _set_export_move(value):
        nonlocal move
        move = value

    return Component(readers={
        "__name__": lambda: __name__,
        'MIN_CURSOR_PIXELS': lambda: MIN_CURSOR_PIXELS,
        'PIXELS_PER_STOP': lambda: PIXELS_PER_STOP,
        '_Probe': lambda: _Probe,
        '_accept': lambda: _accept,
        '_boundary': lambda: _boundary,
        '_candidate': lambda: _candidate,
        '_crossing': lambda: _crossing,
        '_energy': lambda: _energy,
        '_enter': lambda: _enter,
        '_point': lambda: _point,
        '_position': lambda: _position,
        '_probe': lambda: _probe,
        '_relative_energy': lambda: _relative_energy,
        '_stops': lambda: _stops,
        '_stops_cursor': lambda: _stops_cursor,
        '_stops_exit_bracket': lambda: _stops_exit_bracket,
        'color': lambda: color,
        'controls': lambda: controls,
        'dataclass': lambda: dataclass,
        'math': lambda: math,
        'move': lambda: move,
        'np': lambda: np,
        'replace': lambda: replace,
        'reset': lambda: reset,
    }, writers={
        'MIN_CURSOR_PIXELS': _set_export_MIN_CURSOR_PIXELS,
        'PIXELS_PER_STOP': _set_export_PIXELS_PER_STOP,
    '_candidate': _set_export__candidate, '_point': _set_export__point, '_energy': _set_export__energy, '_relative_energy': _set_export__relative_energy, '_Probe': _set_export__Probe, '_probe': _set_export__probe, '_accept': _set_export__accept, '_crossing': _set_export__crossing, 'reset': _set_export_reset, '_stops_cursor': _set_export__stops_cursor, '_position': _set_export__position, '_boundary': _set_export__boundary, '_enter': _set_export__enter, '_stops': _set_export__stops, '_stops_exit_bracket': _set_export__stops_exit_bracket, 'move': _set_export_move})
