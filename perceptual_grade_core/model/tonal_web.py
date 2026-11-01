"""Shared tonal six-handle web in signed modCAM16 coordinates.

Geometry is prepared from Rec.709 RGB rays at the four target tone anchors.
Per-pixel evaluation interpolates relative controls, never compresses coordinates
into a gamut shape, and carries signed energy through both radial operations.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
from ...color_core.modcam16 import HK, State, signed_hk
from ...color_core.scaled import Scalar as S, norm, total

HANDLE_NAMES = ('red', 'yellow', 'green', 'cyan', 'blue', 'magenta')
ZERO46 = ((0.,)*6,)*4
_RAYS = np.array(((1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 1, 1), (0, 0, 1), (1, 0, 1)), dtype=float)
_TWO_PI = 2*np.pi
_CAT16 = np.array([[.401288, .650173, -.051461], [-.250268, 1.204414, .045854],
                   [-.002079, .048952, .953127]])


def cardinal_values(h, anchors, values):
    """C² cardinal tonal interpolation, constant outside the endpoint anchors."""
    h, anchors, values = np.asarray(h, float), np.asarray(anchors, float), np.asarray(values, float)
    index = np.clip(np.searchsorted(anchors, h, side='right')-1, 0, len(anchors)-2)
    u = smoother((h-anchors[index])/(anchors[index+1]-anchors[index]))
    for _ in range(values.ndim-1):
        u = u[..., None]
    return values[index]+(values[index+1]-values[index])*u



@dataclass(frozen=True)
class WebSettings:
    """Relative controls: four tonal rows, six cyclic Rec.709 handles."""
    log_chroma: tuple = ZERO46
    hue_degrees: tuple = ZERO46
    log_scale: tuple = ZERO46

    def __post_init__(self):
        for name in ('log_chroma', 'hue_degrees', 'log_scale'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (4, 6) or not np.isfinite(value).all():
                raise ValueError(f'Web {name} must be finite with shape (4, 6)')
            object.__setattr__(self, name, tuple(tuple(map(float, row)) for row in value))

    @property
    def active(self):
        return any(np.any(getattr(self, n)) for n in
                   ('log_chroma', 'hue_degrees', 'log_scale'))


@dataclass(frozen=True)
class WebGeometry:
    anchors: np.ndarray
    q: np.ndarray
    energy: np.ndarray
    radii: np.ndarray
    angles: np.ndarray


def smoother(t):
    t = np.clip(np.asarray(t, dtype=float), 0, 1)
    return t**3*(10+t*(-15+6*t))


def radial_attenuation(c, toe):
    """(1+(C/toe)^4)^(-1/4), with bounded intermediates."""
    c, toe = np.broadcast_arrays(np.asarray(c, float), np.asarray(toe, float))
    scale = np.maximum(c, toe)
    return (toe/scale)/np.sqrt(np.hypot((c/scale)**2, (toe/scale)**2))


def radial_complement(c, toe):
    """1-A without losing the fourth-order toe term before a large gain."""
    c, toe = np.broadcast_arrays(np.asarray(c, float), np.asarray(toe, float))
    ratio = np.minimum(c, toe)/toe
    fourth = ratio**4
    w = (1+fourth)**.25
    small = fourth/(w*(w+1)*(w*w+1))
    return np.where(c <= toe, small, 1-radial_attenuation(c, toe))


def radial_map(c, g, toe):
    a = radial_attenuation(c, toe)
    g = np.asarray(g)
    gain = np.where(g >= 1, 1+(g-1)*radial_complement(c, toe), g+(1-g)*a)
    return np.asarray(c)*gain


def radial_derivative(c, g, toe):
    a = radial_attenuation(c, toe)
    g = np.asarray(g)
    complement = radial_complement(c, toe)*(1+a+a*a+a**3+a**4)
    return np.where(g >= 1, 1+(g-1)*complement, g+(1-g)*a**5)


def radial_coefficient(source, target, toe):
    a = radial_attenuation(source, toe)
    ratio = np.asarray(target)/source
    return np.where(ratio >= 1, 1+(ratio-1)/(1-a), (ratio-a)/(1-a))


def _soft_min(values):
    values = np.asarray(values, float)
    minimum = np.min(values, axis=-1)
    return minimum/np.sum((minimum[..., None]/values)**8, axis=-1)**.125


def _toe(*radii):
    values = np.concatenate([.1*np.asarray(v) for v in radii], axis=-1)
    return _soft_min(np.concatenate((np.full(values.shape[:-1]+(1,), 2.), values), axis=-1))


def _rec709_rays_at_height(rays, h, reference, encoder=None):
    if encoder is None:
        from ...color_core.modcam16 import encode_cones as encoder
    primaries = np.array([[.64, .33], [.30, .60], [.15, .06]])
    columns = np.vstack((primaries[:, 0], primaries[:, 1], 1-primaries.sum(axis=1)))
    white = np.array([.3127/.329, 1., (1-.3127-.329)/.329])
    matrix = columns*np.linalg.solve(columns, white)[None, :]
    cones = np.asarray(rays) @ (_CAT16 @ matrix).T/(_CAT16 @ white)
    heights = np.broadcast_to(np.asarray(h, float), cones.shape[:-1])
    if np.any(heights <= 0) or not np.isfinite(heights).all():
        raise ValueError('Web geometry requires finite positive target heights')
    low, high = np.zeros_like(heights), np.ones_like(heights)
    for _ in range(64):
        below = encoder(cones*high[..., None], reference).q[..., 0] < heights
        if not np.any(below):
            break
        high = np.where(below, high*2, high)
    else:
        raise ValueError('Rec.709 rays cannot reach requested web height')
    for _ in range(64):
        middle = (low+high)/2
        below = encoder(cones*middle[..., None], reference).q[..., 0] < heights
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    appearance = encoder(cones*((low+high)/2)[..., None], reference)
    return State(appearance.q, appearance.ja*np.abs(appearance.ja))


def rec709_contour(h, reference=203., samples_per_edge=64, encoder=None):
    """Actual fixed-H Rec.709 positive-cone contour, ordered R,Y,G,C,B,M.

    RGB perimeter rays have min(channel)=0, max(channel)=1 before exposure.
    Exposure is free: these contours do not impose an RGB upper channel bound.
    """
    if samples_per_edge < 1:
        raise ValueError('Contour requires at least one sample per edge')
    u = np.arange(samples_per_edge)/samples_per_edge
    rays = (_RAYS[:, None, :]*(1-u[None, :, None])
            +np.roll(_RAYS, -1, axis=0)[:, None, :]*u[None, :, None])
    return _rec709_rays_at_height(rays.reshape(-1, 3), h, reference, encoder)


@lru_cache(maxsize=16)
def _geometry(anchors, reference, encoder=None):
    anchors = np.asarray(anchors, float)
    state = _rec709_rays_at_height(np.broadcast_to(_RAYS, (4, 6, 3)),
                                   anchors[:, None], reference, encoder)
    radii = np.hypot(state.q[..., 1], state.q[..., 2])
    angles = np.unwrap(np.arctan2(state.q[..., 2], state.q[..., 1]), axis=-1)
    if np.any(np.diff(angles, axis=-1) <= 0) or np.any(angles[:, -1] >= angles[:, 0]+_TWO_PI):
        raise ValueError('Rec.709 handles do not have the required cyclic hue order')
    for array in (anchors, state.q, state.energy, radii, angles):
        array.flags.writeable = False
    return WebGeometry(anchors, state.q, state.energy, radii, angles)


def _gather(values, indices):
    return np.take_along_axis(values, np.asarray(indices)[..., None], axis=-1)[..., 0]


def _sector(angle, knots):
    local = np.remainder(angle-knots[..., 0], _TWO_PI)+knots[..., 0]
    extended = np.concatenate((knots, knots[..., :1]+_TWO_PI), axis=-1)
    index = np.minimum(np.sum(local[..., None] >= extended[..., 1:], axis=-1), 5)
    left, right = _gather(extended, index), _gather(extended, index+1)
    return index, (local-left)/(right-left), right-left, local


def _cyclic_field(angle, knots, values):
    index, t, _, _ = _sector(angle, knots)
    extended = np.concatenate((values, values[..., :1]), axis=-1)
    a, b = _gather(extended, index), _gather(extended, index+1)
    return a+(b-a)*smoother(t)


def angular_map(angle, source, target):
    """Monotone periodic quintic, returned on the input angle's lift."""
    index, t, width, local = _sector(angle, source)
    src = np.concatenate((source, source[..., :1]+_TWO_PI), axis=-1)
    dst = np.concatenate((target, target[..., :1]+_TWO_PI), axis=-1)
    slopes = np.diff(dst, axis=-1)/np.diff(src, axis=-1)
    left_slope = np.roll(slopes, 1, axis=-1)
    small = np.minimum(slopes, left_slope)
    derivatives = small/(.5*((small/slopes)**4+(small/left_slope)**4))**.25
    derivatives = np.concatenate((derivatives, derivatives[..., :1]), axis=-1)
    a, b = _gather(dst, index), _gather(dst, index+1)
    da, db = width*_gather(derivatives, index)/5, width*_gather(derivatives, index+1)/5
    values = [a, a+da, a+2*da, b-2*db, b-db, b]
    for _ in range(5):
        values = [(1-t)*x+t*y for x, y in zip(values[:-1], values[1:])]
    return np.where(np.all(source == target, axis=-1), angle, angle+(values[0]-local))


def _energy_at_fixed_h(energy, old_c, new_c):
    """Carry E through a fixed-H radial change without squaring sqrt(E)."""
    b0, b = np.sqrt(HK*old_c), np.sqrt(HK*new_c)
    denominator = np.hypot(np.sqrt(np.abs(energy)), b0)+b0
    u0 = np.divide(energy, denominator, out=np.zeros_like(energy), where=denominator > 0)
    db = np.divide(HK*(new_c-old_c), b+b0, out=np.zeros_like(old_c), where=b+b0 > 0)
    u, du = u0-db, -db
    r = np.abs(u)+np.abs(u0)
    opposite = np.divide(u*u+u0*u0, r, out=np.zeros_like(r), where=r > 0)
    slope = np.where(u*u0 >= 0, r, opposite)
    change = du*(slope+2*b)+2*u0*db
    return np.where(np.abs(change) <= .25*np.abs(energy), energy+change, u*(np.abs(u)+2*b))


def _scaled_energy_at_fixed_h(energy, old_c, new_c):
    b0, b = (HK*old_c)**.5, (HK*new_c)**.5
    denominator = norm([abs(energy)**.5, b0])+b0
    u0 = energy/denominator if denominator else S()
    db = HK*(new_c-old_c)/(b+b0) if b+b0 else S()
    u = u0-db
    r = abs(u)+abs(u0)
    slope = r if u*u0 >= 0 else (u*u+u0*u0)/r
    change = total([-db*(slope+2*b), 2*u0*db])
    return energy+change if abs(change) <= .25*abs(energy) else u*(abs(u)+2*b)


class TonalWeb:
    def __init__(self, settings, tone, encoder=None):
        self.settings, self.tone, self.encoder = settings, tone, encoder
        if not isinstance(settings, WebSettings):
            raise ValueError('Tonal web requires WebSettings')
        self.active = settings.active
        self._geometry = None
        if self.active:
            self.parameters(self.geometry.anchors)

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = _geometry(tuple(100*np.asarray(self.tone.target[1:])), self.tone.reference, self.encoder)
        return self._geometry

    def parameters(self, h):
        h = np.asarray(h, float)
        if not np.isfinite(h).all():
            raise ValueError('Web tonal height must be finite')
        anchors = self.geometry.anchors
        index = np.clip(np.searchsorted(anchors, h, side='right')-1, 0, 2)
        u = smoother((h-anchors[index])/(anchors[index+1]-anchors[index]))
        def blend(v):
            v = np.asarray(v)
            return v[index]+(v[index+1]-v[index])*u[..., None]
        angles = blend(self.geometry.angles)
        radii = np.exp(blend(np.log(self.geometry.radii)))
        target_angles = angles+np.deg2rad(blend(self.settings.hue_degrees))
        with np.errstate(over='ignore', under='ignore'):
            target_radii = radii*np.exp2(blend(self.settings.log_chroma))
            scales = np.exp2(blend(self.settings.log_scale))
            final_radii = target_radii*scales
        gaps = np.diff(np.concatenate((target_angles, target_angles[..., :1]+_TWO_PI), axis=-1), axis=-1)
        if np.any(gaps <= 1.e-8) or not np.isfinite(target_angles).all():
            raise ValueError('Web target hues must preserve strict cyclic order')
        if any(np.any(v <= 0) or not np.isfinite(v).all() for v in (radii, target_radii, scales, final_radii)):
            raise ValueError('Web target radii and scales must be finite and positive')
        angular_toe = _toe(radii)
        chroma_toe = _toe(radii, target_radii)
        scale_toe = _toe(target_radii, final_radii)
        chroma_coefficients = radial_coefficient(radii, target_radii, chroma_toe[..., None])
        scale_coefficients = radial_coefficient(target_radii, final_radii, scale_toe[..., None])
        if any(np.any(v <= 0) or not np.isfinite(v).all() for v in
               (angular_toe, chroma_toe, scale_toe, chroma_coefficients, scale_coefficients)):
            raise ValueError('Web controls exceed the finite geometry coefficient range')
        return dict(angles=angles, radii=radii, target_angles=target_angles,
                    target_radii=target_radii, scales=scales, angular_toe=angular_toe,
                    chroma_toe=chroma_toe, scale_toe=scale_toe,
                    chroma_coefficients=chroma_coefficients, scale_coefficients=scale_coefficients)

    def settings_for_target(self, tonal_row, handle, target_ab, offset=None):
        """Convert a combined final a/b target, retaining scale.

        With a frozen Adaptance offset, subtract its scaled output contribution
        before deriving the source-centered web target. The hue is lifted closest to the current target before order validation.
        Rejected crossed/collapsed handles leave the immutable settings unchanged.
        """
        if not 0 <= tonal_row < 4 or not 0 <= handle < 6:
            raise ValueError('Web handle index is outside its four-by-six grid')
        target = np.asarray(target_ab, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Web drag target must be a finite a/b pair')
        p = self.parameters(self.geometry.anchors[tonal_row])
        if offset is not None:
            # Final cursor = lambda * (pre-offset web target + frozen offset).
            offset = np.asarray(offset, float)
            if offset.shape != (2,) or not np.isfinite(offset).all():
                raise ValueError('Web target offset must be a finite a/b pair')
            target = target-p['scales'][handle]*offset
        radius = float(np.hypot(*target))
        if radius <= 0:
            raise ValueError('Web handles cannot collapse onto neutral')
        p = self.parameters(self.geometry.anchors[tonal_row])
        previous = p['target_angles'][handle]
        angle = np.arctan2(target[1], target[0])
        angle = previous+np.remainder(angle-previous+np.pi, _TWO_PI)-np.pi
        chroma = np.array(self.settings.log_chroma, dtype=float)
        hue = np.array(self.settings.hue_degrees, dtype=float)
        chroma[tonal_row, handle] = np.log2(radius/p['radii'][handle])-self.settings.log_scale[tonal_row][handle]
        hue[tonal_row, handle] = np.rad2deg(angle-p['angles'][handle])
        settings = replace(self.settings, log_chroma=tuple(map(tuple, chroma)),
                           hue_degrees=tuple(map(tuple, hue)))
        TonalWeb(settings, self.tone, self.encoder)  # Validate the complete target ordering.
        return settings

    def apply(self, q, energy, tonal_h, offset=None, *, transform=None):
        """Apply hue/chroma, an optional appearance transform, then final scale.

        The final scale is selected on the hue/chroma result before the
        transform. ``offset`` remains available for stand-alone web callers.
        """
        q = np.asarray(q, float)
        if q.shape[-1:] != (3,) or not np.isfinite(q).all():
            raise ValueError('Web coordinates must be finite triples')
        energy = np.broadcast_to(np.asarray(energy, float), q.shape[:-1])
        tonal_h = np.broadcast_to(np.asarray(tonal_h, float), q.shape[:-1])
        if not np.isfinite(energy).all() or not np.isfinite(tonal_h).all():
            raise ValueError('Web energy and tonal height must be finite')
        offset = np.zeros(q.shape[:-1]+(2,)) if offset is None else np.broadcast_to(np.asarray(offset, float), q.shape[:-1]+(2,))
        if not np.isfinite(offset).all():
            raise ValueError('Web offset must be finite')
        if not self.active:
            result = q.copy()
            result[..., 1:] += offset
            c0 = np.hypot(q[..., 1], q[..., 2])
            c1 = np.hypot(result[..., 1], result[..., 2])
            out_energy = _energy_at_fixed_h(energy, c0, c1)
            return (result, out_energy) if transform is None else transform(result, out_energy)
        p = self.parameters(tonal_h)
        c = np.hypot(q[..., 1], q[..., 2])
        angle = np.arctan2(q[..., 2], q[..., 1])
        warped = angular_map(angle, p['angles'], p['target_angles'])
        out_angle = angle+smoother(c/p['angular_toe'])*(warped-angle)
        gc = np.exp(_cyclic_field(out_angle, p['target_angles'], np.log(p['chroma_coefficients'])))
        c1 = radial_map(c, gc, p['chroma_toe'])
        e1 = _energy_at_fixed_h(energy, c, c1)
        gs = np.exp(_cyclic_field(out_angle, p['target_angles'], np.log(p['scale_coefficients'])))
        c2 = radial_map(c1, gs, p['scale_toe'])
        scale = np.divide(c2, c1, out=np.ones_like(c1), where=c1 > 0)
        radial_gain = np.divide(c1, c, out=np.ones_like(c), where=c > 0)
        polar_ab = np.stack((c1*np.cos(out_angle), c1*np.sin(out_angle)), axis=-1)
        ab1 = np.where((out_angle == angle)[..., None], q[..., 1:]*radial_gain[..., None], polar_ab)
        shifted = ab1+offset
        shifted_c = np.where(np.all(offset == 0, axis=-1), c1,
                             np.hypot(shifted[..., 0], shifted[..., 1]))
        shifted_e = _energy_at_fixed_h(e1, c1, shifted_c)
        transformed = np.concatenate((q[..., :1], shifted), axis=-1)
        if transform is not None:
            transformed, shifted_e = transform(transformed, shifted_e)
            shifted = transformed[..., 1:]
            shifted_c = np.hypot(shifted[..., 0], shifted[..., 1])
        e2 = shifted_e*scale*scale
        ab = shifted*scale[..., None]
        final_c = shifted_c*scale
        h2 = signed_hk(np.sign(e2)*np.sqrt(np.abs(e2)), final_c)
        h2 = np.where(scale == 1, transformed[..., 0], h2)
        result = np.concatenate((h2[..., None], ab), axis=-1)
        unchanged = (out_angle == angle) & (c2 == c) & (scale == 1) & np.all(offset == 0, axis=-1) & (transform is None)
        result = np.where(unchanged[..., None], q, result)
        e2 = np.where(unchanged & (scale == 1), energy, e2)
        return result, e2

    def scaled_apply(self, q, energy, tonal_h, offset=None, *, transform=None):
        q, energy, tonal_h = list(map(S, q)), S(energy), S(tonal_h)
        if len(q) != 3:
            raise ValueError('Web coordinates must have three components')
        offset = [S(), S()] if offset is None else list(map(S, offset))
        if len(offset) != 2:
            raise ValueError('Web offset must have two components')
        if not self.active:
            result = [q[0], *(v+d for v, d in zip(q[1:], offset))]
            out_energy = _scaled_energy_at_fixed_h(energy, norm(q[1:]), norm(result[1:]))
            return (result, out_energy) if transform is None else transform(result, out_energy)
        h = min(S(self.geometry.anchors[-1]), max(S(self.geometry.anchors[0]), S(tonal_h)))
        p = self.parameters(float(h))
        c = norm(q[1:])
        if not c:
            result = [q[0], *offset]
            out_energy = _scaled_energy_at_fixed_h(energy, c, norm(offset))
            return (result, out_energy) if transform is None else transform(result, out_energy)
        angle = np.arctan2(float(q[2]/c), float(q[1]/c))
        warped = float(angular_map(angle, p['angles'], p['target_angles']))
        t = min(S(1.), c/p['angular_toe'])
        fade = t**3*(10+t*(-15+6*t))
        out_angle = angle+float(fade)*(warped-angle)
        gc = float(np.exp(_cyclic_field(out_angle, p['target_angles'], np.log(p['chroma_coefficients']))))
        def radial(value, g, toe):
            fourth = (value/toe)**4
            w = (1+fourth)**.25
            complement = fourth/(w*(w+1)*(w*w+1))
            a = 1/w
            return value*(1+(g-1)*complement if g >= 1 else g+(1-g)*a)
        c1 = radial(c, gc, p['chroma_toe'])
        e1 = _scaled_energy_at_fixed_h(energy, c, c1)
        gs = float(np.exp(_cyclic_field(out_angle, p['target_angles'], np.log(p['scale_coefficients']))))
        c2 = radial(c1, gs, p['scale_toe'])
        scale = c2/c1 if c1 else S(1.)
        ab1 = ([v*(c1/c) for v in q[1:]] if out_angle == angle
               else [c1*np.cos(out_angle), c1*np.sin(out_angle)])
        shifted = [v+d for v, d in zip(ab1, offset)]
        shifted_c = norm(shifted) if any(offset) else c1
        shifted_e = _scaled_energy_at_fixed_h(e1, c1, shifted_c)
        transformed = [q[0], *shifted]
        if transform is not None:
            transformed, shifted_e = transform(transformed, shifted_e)
            shifted = transformed[1:]
            shifted_c = norm(shifted)
        e2 = shifted_e*scale*scale
        ja = (-1 if e2 < 0 else 1)*abs(e2)**.5
        h2 = transformed[0] if scale == 1 else signed_hk(ja, shifted_c*scale)
        if transform is None and out_angle == angle and c2 == c and scale == 1 and not any(offset):
            return q, energy
        return [h2, *(v*scale for v in shifted)], e2
