"""Selective cone-derived tint, with neutral achromatic normalization.

Physical responses use exact conjugated cone gains. The exterior continuation
is algebraic, not physical cone uncompression. Export guards remain necessary.
"""
from dataclasses import replace
import numpy as np
from ...color_core import modcam16 as default_response_math
from ...color_core.modcam16 import HK
from ...color_core.scaled import Scalar as S, dot, norm, signed_power
from .tonal_web import cardinal_values, _energy_at_fixed_h
from .saturation import project, bounded_project, bounded_unproject
from .hk_guard import guard

B = 400.
_WEIGHTS = np.array([2., 1., .05])


def response_transform(response, gain):
    """Total monotone response map; inverse is the reciprocal gain."""
    r, m = np.broadcast_arrays(np.asarray(response, float), np.asarray(gain, float))
    if np.any(m <= 0) or not np.isfinite(m).all():
        raise ValueError('Response gains must be finite and positive')
    t = np.minimum(abs(r)/B, 1.)
    result = r*(m/((1-t)+m*t))
    outside = abs(r) > B
    if np.any(outside):
        y = abs(r[outside])/B-1
        root_y = np.sqrt(y)
        u = root_y*(2*root_y/(np.hypot(1., 2*root_y)+1))
        z = u/m[outside]
        result = np.array(result, copy=True)
        result[outside] = np.sign(r[outside])*B*(1+z+z*z)
    return np.where(m == 1., r, result)


def response_transform_scaled(response, gain):
    r, m = S(response), S(gain)
    if m <= 0:
        raise ValueError('Response gains must be positive')
    if m == 1 or not r:
        return r
    if abs(r) <= B:
        t = abs(r)/B
        return m*r/((1-t)+m*t)
    y = abs(r)/B-1
    u = 2*y/((1+4*y)**.5+1)
    z = u/m
    return (-1 if r < 0 else 1)*B*(1+z+z*z)


def reach_weight(chroma, energy, half_saturation):
    """Fourth-power mask, half active at C/|J_A| = half_saturation."""
    c, e = np.broadcast_arrays(np.asarray(chroma, float), np.asarray(energy, float))
    a = half_saturation*np.sqrt(abs(e))
    scale = np.maximum(a, c)
    an = np.divide(a, scale, out=np.zeros_like(a), where=scale > 0)
    cn = np.divide(c, scale, out=np.zeros_like(c), where=scale > 0)
    return np.divide(an**4, an**4+cn**4, out=np.ones_like(a), where=scale > 0)


def reach_weight_scaled(chroma, energy, half_saturation):
    a = S(half_saturation)*abs(S(energy))**.5
    c = S(chroma)
    if not c:
        return 1.
    if not a:
        return 0.
    scale = max(a, c)
    aa, cc = (a/scale)**4, (c/scale)**4
    return float(aa/(aa+cc))


def to_responses(q, energy, aw, *, response_math=default_response_math):
    q = np.asarray(q, float)
    angle = np.arctan2(q[..., 2], q[..., 1])
    opponent = q[..., 1:]*(aw/(1204*response_math.eccentricity(angle)))[..., None]
    a = response_math.achromatic_response(energy, aw)
    return np.concatenate((a[..., None], opponent), axis=-1) @ response_math.INVERSE.T


def from_responses(response, aw, *, response_math=default_response_math):
    r = np.asarray(response, float)
    a = r @ _WEIGHTS
    # Difference forms cancel a common neutral without row-sum rounding.
    u = (r[..., 0]-r[..., 1])+(r[..., 2]-r[..., 1])/11
    w = ((r[..., 0]-r[..., 2])+(r[..., 1]-r[..., 2]))/9
    angle = np.arctan2(w, u)
    ab = np.stack((u, w), axis=-1)*(1204*response_math.eccentricity(angle)/aw)[..., None]
    ja = 100*signed_power(a/aw, response_math.GAMMA)
    e = ja*abs(ja)
    c = np.hypot(ab[..., 0], ab[..., 1])
    return np.concatenate((response_math.signed_hk(ja, c)[..., None], ab), axis=-1), e


class SelectiveAdaptance:
    """Four direct neutral destinations, smoothly interpolated cone log gains."""
    def __init__(self, targets, anchors, aw, half_saturation=.25, *, response_math=default_response_math):
        self.response_math = response_math
        self.targets, self.anchors = np.asarray(targets, float), np.asarray(anchors, float)
        self.aw, self.half_saturation = float(aw), float(half_saturation)
        if (self.targets.shape != (4, 2) or not np.isfinite(self.targets).all()
                or self.anchors.shape != (4,) or not np.isfinite(self.anchors).all()
                or np.any(self.anchors <= 0) or np.any(np.diff(self.anchors) <= 0)
                or not np.isfinite(self.aw) or self.aw <= 0
                or not np.isfinite(self.half_saturation) or not .05 <= self.half_saturation <= 1):
            raise ValueError('Invalid selective Adaptance targets, anchors or half-influence saturation')
        self.active = bool(np.any(self.targets))
        r0 = self.aw/3.05*(self.anchors/100)**(1/self.response_math.GAMMA)
        if np.any(r0 >= B):
            raise ValueError('Adaptance calibration anchors exceed physical response range')
        angle = np.arctan2(self.targets[:, 1], self.targets[:, 0])
        opponent = self.targets*(self.aw/(1204*self.response_math.eccentricity(angle)))[:, None]
        target = r0[:, None]+opponent @ self.response_math.INVERSE[:, 1:].T
        if np.any(target <= 0) or np.any(target >= B):
            raise ValueError('Neutral target requires positive cone responses below 400')
        # Log odds avoid overflowing calibrated gain quotients.
        logs = np.log(target)-np.log(B-target)-np.log(r0[:, None])+np.log(B-r0[:, None])
        self.log_gains = np.where(np.all(self.targets == 0, axis=-1)[:, None], 0., logs)

    def parameters(self, h):
        logs = cardinal_values(h, self.anchors, self.log_gains)
        m = np.exp(logs)
        r0 = self.aw/3.05*(np.asarray(h)/100)**(1/self.response_math.GAMMA)
        # Inner quotient has an analytic limit at black; exterior stays total.
        t = np.minimum(r0/B, 1.)[..., None]
        ratio = m/((1-t)+m*t)
        outside = r0 > B
        if np.any(outside):
            ratio = np.array(ratio, copy=True)
            ratio[outside] = response_transform(r0[outside, None], m[outside])/r0[outside, None]
        k = 3.05/(ratio @ _WEIGHTS)
        return logs, k

    def apply(self, q, energy, h, source_chroma, source_energy):
        q, energy = np.asarray(q, float), np.asarray(energy, float)
        if not self.active:
            return q.copy(), energy.copy()
        logs, k = self.parameters(h)
        v = reach_weight(source_chroma, source_energy, self.half_saturation)
        gains = np.exp(v[..., None]*logs)
        factor = np.exp(v*np.log(k))
        responses = response_transform(to_responses(q, energy, self.aw, response_math=self.response_math), gains)*factor[..., None]
        out, e = from_responses(responses, self.aw, response_math=self.response_math)
        neutral = (np.asarray(source_chroma) == 0) & np.all(q[..., 1:] == 0, axis=-1)
        e = np.where(neutral, energy, e)
        out[..., 0] = self.response_math.signed_hk(np.sign(e)*np.sqrt(abs(e)), np.hypot(out[..., 1], out[..., 2]))
        unchanged = ((v == 0) | (np.all(gains == 1, axis=-1) & (factor == 1))
                     | ((energy == 0) & np.all(q == 0, axis=-1)))
        return np.where(unchanged[..., None], q, out), np.where(unchanged, energy, e)

    def inverse(self, q, energy, h, source_chroma, source_energy):
        """Explicit inverse with supplied, fixed original selectors."""
        if not self.active:
            return np.array(q, copy=True), np.array(energy, copy=True)
        logs, k = self.parameters(h)
        v = reach_weight(source_chroma, source_energy, self.half_saturation)
        responses = to_responses(q, energy, self.aw, response_math=self.response_math)/np.exp(v*np.log(k))[..., None]
        return from_responses(response_transform(responses, np.exp(-v[..., None]*logs)), self.aw, response_math=self.response_math)

    def scaled_apply(self, q, energy, h, source_chroma, source_energy):
        q, energy, h = list(map(S, q)), S(energy), S(h)
        if not self.active or (not energy and not any(q)):
            return q, energy
        selector = float(min(S(self.anchors[-1]), max(S(self.anchors[0]), h)))
        logs = cardinal_values(selector, self.anchors, self.log_gains)
        m = np.exp(logs)
        r0 = self.aw/3.05*(h/100)**(1/self.response_math.GAMMA)
        ratios = [response_transform_scaled(r0, g)/r0 if r0 else S(g) for g in m]
        k = S(3.05)/dot(_WEIGHTS, ratios)
        v = reach_weight_scaled(source_chroma, source_energy, self.half_saturation)
        gains = np.exp(v*logs)
        factor = k**v
        if not v or (all(g == 1 for g in gains) and factor == 1):
            return q, energy
        c = norm(q[1:])
        angle = np.arctan2(float(q[2]/c), float(q[1]/c)) if c else 0.
        components = [self.response_math.achromatic_response(energy, self.aw), *(z*self.aw/(1204*float(self.response_math.eccentricity(angle))) for z in q[1:])]
        responses = [factor*response_transform_scaled(dot(row, components), g) for row, g in zip(self.response_math.INVERSE, gains)]
        a = dot(_WEIGHTS, responses)
        u = (responses[0]-responses[1])+(responses[2]-responses[1])/11
        w = ((responses[0]-responses[2])+(responses[1]-responses[2]))/9
        radius = norm([u, w])
        angle = np.arctan2(float(w/radius), float(u/radius)) if radius else 0.
        ab = [z*(1204*float(self.response_math.eccentricity(angle))/self.aw) for z in (u, w)]
        ja = 100*signed_power(a/self.aw, self.response_math.GAMMA)
        e = energy if not S(source_chroma) and not c else ja*abs(ja)
        ja = signed_power(e, .5)
        return [self.response_math.signed_hk(ja, norm(ab)), *ab], e

    def target(self, web, row, handle, target_ab):
        """Recover an underlying web target using a local control-only solve."""
        return self._target(web, row, handle, target_ab, saturation=False)

    def saturation_target(self, web, row, handle, target):
        """Recover a saturation target without crossing the current chart branch.

        Final Scale cancels from the chart. Both carried-energy signs and the
        local derivative orientation are held at their seed values. An invalid
        request raises before publishing new immutable settings.
        """
        return self._target(web, row, handle, target, saturation=True)

    def _target(self, web, row, handle, target_ab, *, saturation):
        p = web.parameters(web.geometry.anchors[row])
        h = web.geometry.anchors[row]
        source_c = web.geometry.radii[row, handle]
        source_e = web.geometry.energy[row, handle]
        scale = p['scales'][handle]
        angle = p['target_angles'][handle]
        z = p['target_radii'][handle]*np.array([np.cos(angle), np.sin(angle)])
        target = np.asarray(target_ab, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Adapted corner target must be a finite a/b pair')
        tolerance = 5.e-14*max(1., float(np.max(abs(target))))
        branch = None
        orientation = None
        def evaluate(ab):
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    c = np.hypot(*ab)
                    e = _energy_at_fixed_h(np.asarray(source_e), np.asarray(source_c), np.asarray(c))
                    q = np.array([self.response_math.signed_hk(np.sign(e)*np.sqrt(abs(e)), c), *ab])
                    adapted, adapted_e = self.apply(q, e, h, source_c, source_e)
                    signs = (np.sign(e), np.sign(adapted_e))
                    if saturation and (0 in signs or (branch is not None and signs != branch)):
                        raise ValueError('Saturation target crosses its current energy branch')
                    value = project(adapted, adapted_e) if saturation else adapted[1:]*scale
            except FloatingPointError as error:
                raise ValueError('Adapted corner target exceeds the finite solve domain') from error
            if not np.isfinite(value).all():
                raise ValueError('Adapted corner target exceeds the finite solve domain')
            return value, signs

        def derivative(ab):
            step = 1.e-5*max(1., np.hypot(*ab))
            # Close to E=0, finite-difference probes must shrink on the same
            # branch too. Never use a derivative spanning the infinite chart.
            for _ in range(20):
                axes = np.eye(2)*step
                try:
                    jac = np.column_stack([(evaluate(ab+d)[0]-evaluate(ab-d)[0])/(2*step) for d in axes])
                    break
                except ValueError:
                    step *= .5
            else:
                raise ValueError('Adapted corner target exceeds the finite solve domain or has no local derivative')
            if not np.isfinite(jac).all():
                raise ValueError('Adapted corner target has a nonfinite local derivative')
            try:
                sign, _ = np.linalg.slogdet(jac)
                folded = (sign == 0 or np.linalg.cond(jac) > 1.e10 or
                          (sign != orientation if orientation is not None else not saturation and sign < 0))
            except np.linalg.LinAlgError as error:
                raise ValueError('Adapted corner target is locally singular') from error
            if folded:
                raise ValueError('Adapted corner target is locally folded or singular')
            return jac, sign

        _, seed_branch = evaluate(z)
        if saturation:
            branch = seed_branch
        _, orientation = derivative(z)
        for _ in range(32):
            with np.errstate(over='ignore', invalid='ignore'):
                residual = evaluate(z)[0]-target
            if not np.isfinite(residual).all():
                raise ValueError('Adapted corner target has a nonfinite residual')
            jac, _ = derivative(z)
            if np.hypot(*residual) <= tolerance:
                return web.settings_for_target(row, handle, z*scale)
            try:
                delta = np.linalg.solve(jac, residual)
            except np.linalg.LinAlgError as error:
                raise ValueError('Adapted corner target is locally singular') from error
            if not np.isfinite(delta).all():
                raise ValueError('Adapted corner target has a nonfinite solve step')
            if saturation:
                delta /= max(1., np.hypot(*delta)/(.5*max(1., np.hypot(*z))))
            for backtrack in range(16):
                candidate = z-delta*2.**(-backtrack)
                try:
                    candidate_value = evaluate(candidate)[0]
                    if saturation:
                        # A valid endpoint alone does not establish that the
                        # local step stayed on one unfolded chart component.
                        for fraction in (.25, .5, .75, 1.):
                            derivative(z+fraction*(candidate-z))
                except ValueError:
                    continue
                if np.hypot(*(candidate_value-target)) < np.hypot(*residual):
                    z = candidate
                    break
            else:
                raise ValueError('Adapted corner target solve did not improve')
        raise ValueError('Adapted corner target solve did not converge')


class GuardedWarpDisplay:
    """Portable display adapters; the raw grading/Stack path stays untouched.

    The host supplies ``warp_state``, ``warp_targets``, ``web``, ``adaptation``
    and immutable ``settings``. Guard context always belongs to the original
    point, not the destination and not a reconstructed rounded H coordinate.
    """
    def guarded_warp_state(self, state, tonal_h=None, source_chroma=None, source_energy=None):
        raw = self.warp_state(state, tonal_h, source_chroma, source_energy)
        reference = state.energy if source_energy is None else source_energy
        return self.adaptation.response_math.State(*guard(raw.q, raw.energy, reference))

    def guarded_warp_targets(self, tonal_row):
        geometry = self.web.geometry
        raw = self.warp_targets(tonal_row)
        reference = np.r_[geometry.anchors[tonal_row]**2, geometry.energy[tonal_row]]
        return self.adaptation.response_math.State(*guard(raw.q, raw.energy, reference))

    def guarded_saturation_targets(self, tonal_row):
        state = self.guarded_warp_targets(tonal_row)
        return bounded_project(state.q, state.energy)

    def guard_rim_radius(self, tonal_row, handle):
        """Selected corner's E=0 circle at its current final raw H and E0."""
        if not 0 <= handle < 6:
            raise ValueError('An H-K rim requires a corner handle')
        h = float(self.warp_targets(tonal_row).q[handle+1, 0])
        if h <= 0:
            return 0.
        q, e = guard(np.array([h, h*h/HK, 0.]), 0., self.web.geometry.energy[tonal_row, handle])
        return float(np.hypot(*bounded_project(q, e)))

    def guarded_corner_evaluator(self, tonal_row, handle):
        """Freeze a corner context and return (raw State, guarded pad point).

        The callable accepts optional requested log2 chroma gain and absolute
        pre-Adaptance hue in radians. It avoids model construction and global
        geometry validation during root probes; callers validate final settings.
        """
        return _GuardedCorner(self.adaptation, self.web, tonal_row, handle)

    def target_guarded_saturation_settings(self, tonal_row, handle, target):
        target = np.asarray(target, float)
        if target.shape != (2,) or not np.isfinite(target).all() or np.hypot(*target) >= 1:
            raise ValueError('Guarded saturation target must be a finite open-disk pair')
        if not 0 <= tonal_row < 4 or not -1 <= handle < 6:
            raise ValueError('Warp target index is outside its four-by-seven grid')
        if handle == -1:
            # A neutral source retains E=H_source², and therefore its guard is
            # identity even after Adaptance. The exact center inverse survives.
            return self.target_settings(tonal_row, handle,
                                        self.web.geometry.anchors[tonal_row]*bounded_unproject(target))
        settings = self.guarded_corner_evaluator(tonal_row, handle).target(target)
        return replace(self.settings, web=settings)

    def target_guarded_hue_settings(self, tonal_row, handle, log_chroma, angle):
        """Hold requested chroma stops while solving the guarded output hue."""
        settings = self.guarded_corner_evaluator(tonal_row, handle).hue_target(log_chroma, angle)
        return replace(self.settings, web=settings)

    def guarded_boundary_settings(self, tonal_row, handle, angle=None):
        """Target final carried E=0 at the supplied guarded hue direction."""
        settings = self.guarded_corner_evaluator(tonal_row, handle).boundary(angle)
        return replace(self.settings, web=settings)


class _GuardedCorner:
    """Local signed-response solve for a single calibrated web corner."""
    def __init__(self, adaptation, web, row, handle):
        if not 0 <= row < 4 or not 0 <= handle < 6:
            raise ValueError('Warp corner index is outside its four-by-six grid')
        self.adaptation, self.web, self.row, self.handle = adaptation, web, row, handle
        self.h = float(web.geometry.anchors[row])
        self.source_c = float(web.geometry.radii[row, handle])
        self.source_e = float(web.geometry.energy[row, handle])
        p = web.parameters(self.h)
        self.scale = float(p['scales'][handle])
        self.angle = float(p['target_angles'][handle])
        self.log_chroma = float(web.settings.log_chroma[row][handle])
        # With no active cone transform, E itself is the regular coordinate.
        # Active cone maps are regular in signed A, including passage through 0.
        self.power = 2*adaptation.response_math.GAMMA if adaptation.active else 1.

    def _state(self, c, angle, energy=None):
        if not np.isfinite([c, angle]).all() or c <= 0:
            raise ValueError('Guarded corner requires positive finite chroma')
        if energy is None:
            energy = _energy_at_fixed_h(np.asarray(self.source_e), np.asarray(self.source_c), np.asarray(c))
        q = np.array([self.h, c*np.cos(angle), c*np.sin(angle)])
        q, e = self.adaptation.apply(q, np.asarray(energy), self.h, self.source_c, self.source_e)
        q = np.asarray(q, float).copy()
        q[1:] *= self.scale
        e = float(e)*self.scale*self.scale
        if self.scale != 1:
            q[0] = self.adaptation.response_math.signed_hk(np.sign(e)*np.sqrt(abs(e)), np.hypot(*q[1:]))
        if not np.isfinite(q).all() or not np.isfinite(e):
            raise ValueError('Guarded corner exceeds the finite appearance domain')
        raw = self.adaptation.response_math.State(q, np.asarray(e))
        guarded, stored = guard(q, e, self.source_e)
        return raw, bounded_project(guarded, stored)

    def __call__(self, log_chroma=None, angle=None):
        log_chroma = self.log_chroma if log_chroma is None else float(log_chroma)
        angle = self.angle if angle is None else float(angle)
        with np.errstate(over='raise', invalid='raise'):
            try:
                return self._state(self.source_c*np.exp2(log_chroma), angle)
            except FloatingPointError as error:
                raise ValueError('Guarded corner exceeds the finite control domain') from error

    def _coordinates(self, log_chroma=None, angle=None):
        c = self.source_c*np.exp2(self.log_chroma if log_chroma is None else log_chroma)
        e = float(_energy_at_fixed_h(np.asarray(self.source_e), np.asarray(self.source_c), np.asarray(c)))
        normalized = e/(self.h*self.h)
        return np.array([np.sign(normalized)*abs(normalized)**(1/self.power),
                         self.angle if angle is None else angle])

    def _from_coordinates(self, z):
        normalized = np.sign(z[0])*abs(z[0])**self.power
        if not np.isfinite(normalized) or normalized >= 1:
            raise ValueError('Guarded corner cannot collapse onto neutral')
        # Invert the signed fixed-H energy exactly. For E<0, H=2b-sqrt(b²-E).
        c = (self.h*self.h/HK)*(1-normalized if normalized >= 0 else
                                      ((2+np.sqrt(1-3*normalized))/3)**2)
        return c, normalized*self.h*self.h

    def _evaluate(self, z):
        c, e = self._from_coordinates(z)
        raw, point = self._state(c, z[1], e)
        if raw.q[0] <= 0 or not np.isfinite(point).all():
            raise ValueError('Guarded destination is outside the positive-H display domain')
        return raw, point

    def _settings(self, z):
        c, _ = self._from_coordinates(z)
        return self.web.settings_for_target(self.row, self.handle,
                 c*self.scale*np.array([np.cos(z[1]), np.sin(z[1])]))

    def _solve(self, residual, seed=None):
        z = self._coordinates() if seed is None else np.array(seed, float)
        orientation = None
        def derivative(at):
            steps = 2.e-5*np.maximum(1., abs(at))
            for _ in range(20):
                try:
                    jac = np.column_stack([(residual(at+np.eye(2)[i]*steps[i])-
                                            residual(at-np.eye(2)[i]*steps[i]))/(2*steps[i])
                                           for i in range(2)])
                    break
                except (ValueError, FloatingPointError):
                    steps *= .5
            else:
                raise ValueError('Guarded target has no finite local derivative')
            try:
                sign, _ = np.linalg.slogdet(jac)
                bad = (not np.isfinite(jac).all() or sign == 0 or np.linalg.cond(jac) > 1.e12
                       or (orientation is not None and sign != orientation))
            except np.linalg.LinAlgError as error:
                raise ValueError('Guarded target is locally singular') from error
            if bad:
                raise ValueError('Guarded target is locally folded or singular')
            return jac, sign
        _, orientation = derivative(z)
        for _ in range(48):
            error = residual(z)
            jac, _ = derivative(z)
            if np.linalg.norm(error) <= 2.e-12:
                return z
            delta = np.linalg.solve(jac, error)
            # Bound both response and angle steps, then validate the whole local
            # step. E signs are deliberately unrestricted in this guarded chart.
            delta /= max(1., abs(delta[0])/(.5*max(1., abs(z[0]))), abs(delta[1])/.5)
            for backtrack in range(20):
                candidate = z-delta*2.**(-backtrack)
                try:
                    value = residual(candidate)
                    if np.linalg.norm(value) >= np.linalg.norm(error):
                        continue
                    for fraction in (.25, .5, .75, 1.):
                        derivative(z+fraction*(candidate-z))
                except (ValueError, FloatingPointError):
                    continue
                if np.linalg.norm(value) < np.linalg.norm(error):
                    z = candidate
                    break
            else:
                raise ValueError('Guarded target solve did not improve')
        raise ValueError('Guarded target solve did not converge')

    def target(self, point):
        point = np.asarray(point, float)
        residual = lambda value: self._evaluate(value)[1]-point
        try:
            return self._settings(self._solve(residual))
        except ValueError as original:
            # Cone Adaptance can produce a microscopic fold at A=0: fixed-H
            # chroma changes as |A|^(2 Gamma), while transformed chroma has a
            # small linear A term. Such lobes may fit within the saved-control
            # reprojection tolerance. Restart only from a subpixel point
            # on the requested side; every subsequent step retains the ordinary
            # fold/conditioning checks, as does the final target derivative.
            seed = self._coordinates()
            if not self.adaptation.active or abs(seed[0]) > .03:
                raise
            initial = self._evaluate(seed)[1]
            direction = 1. if np.linalg.norm(point) < np.linalg.norm(initial) else -1.
            for step in (.0001, .0003, .001, .003, .01, .03):
                candidate = seed.copy()
                candidate[0] += direction*step
                try:
                    c, _ = self._from_coordinates(candidate)
                    angular = self.hue_target(np.log2(c/self.source_c), float(np.arctan2(initial[1], initial[0])))
                    candidate[1] = self.web.geometry.angles[self.row, self.handle]+np.deg2rad(angular.hue_degrees[self.row][self.handle])
                    if np.linalg.norm(self._evaluate(candidate)[1]-initial) > 2.e-5:
                        continue
                    z = self._solve(residual, candidate)
                    return self._settings(z)
                except ValueError:
                    continue
            raise original

    def boundary(self, angle=None):
        if angle is None:
            _, point = self()
            angle = float(np.arctan2(point[1], point[0]))
        if not np.isfinite(angle):
            raise ValueError('Guarded hue must be finite')
        if not self.adaptation.active:
            return self._settings(np.array([0., self._lift(angle, self.angle)]))
        def residual(z):
            raw, point = self._evaluate(z)
            response = np.sign(raw.energy)*abs(raw.energy/(self.h*self.h*self.scale*self.scale))**(1/(2*self.adaptation.response_math.GAMMA))
            direction = np.arctan2(point[1], point[0])
            return np.array([response, self._lift(direction, angle)-angle])
        z = self._solve(residual, seed=[0., self._lift(angle, self.angle)])
        return self._settings(z)

    @staticmethod
    def _lift(angle, reference):
        return reference+np.remainder(angle-reference+np.pi, 2*np.pi)-np.pi

    def hue_target(self, log_chroma, angle):
        if not np.isfinite([log_chroma, angle]).all():
            raise ValueError('Guarded chroma and hue controls must be finite')
        candidate = self.angle
        def residual(value):
            raw, point = self(log_chroma, value)
            if raw.q[0] <= 0:
                raise ValueError('Guarded destination is outside the positive-H display domain')
            return self._lift(float(np.arctan2(point[1], point[0])), angle)-angle
        orientation = None
        def derivative(value):
            step = 1.e-5
            result = (residual(value+step)-residual(value-step))/(2*step)
            if not np.isfinite(result) or abs(result) < 1.e-10 or (orientation is not None and np.sign(result) != orientation):
                raise ValueError('Guarded hue is locally folded or singular')
            return result
        orientation = np.sign(derivative(candidate))
        for _ in range(32):
            error = residual(candidate)
            slope = derivative(candidate)
            if abs(error) < 2.e-12:
                c = self.source_c*np.exp2(log_chroma)
                return self.web.settings_for_target(self.row, self.handle,
                       c*self.scale*np.array([np.cos(candidate), np.sin(candidate)]))
            delta = np.clip(error/slope, -.5, .5)
            for backtrack in range(16):
                value = candidate-delta*2.**(-backtrack)
                try:
                    new_error = residual(value)
                    for fraction in (.25, .5, .75, 1.):
                        derivative(candidate+fraction*(value-candidate))
                except ValueError:
                    continue
                if abs(new_error) < abs(error):
                    candidate = value
                    break
            else:
                raise ValueError('Guarded hue solve did not improve')
        raise ValueError('Guarded hue solve did not converge')
