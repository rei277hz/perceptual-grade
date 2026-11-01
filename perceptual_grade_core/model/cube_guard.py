"""Validated fixed-J_A, fixed-hue projection onto the native AP1 cube.

The fitted capacity is only a search seed.  Every accepted radius is checked
through the actual appearance decoder and native RGB adapter.  The tested RGB
is returned with the appearance state so export need not decode it again.
"""
import numpy as np

from ...color_core.modcam16 import GAMMA, INVERSE, eccentricity, signed_hk
from ...color_core.scaled import Scalar as S, norm

MAXIMUM = 1024.
FLOAT64_STEPS = 56


def _decoded(context, q, energy):
    """Return the actual legal RGB, or None (including a response-pole trial)."""
    try:
        response = context.responses(np.asarray(q)[None, :], np.asarray([energy]))
        if not np.isfinite(response).all() or np.any(abs(response) >= 400.):
            return None
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            rgb = np.asarray(context.decode(np.asarray(q)[None, :], np.asarray([energy]))[0])
    except (ValueError, FloatingPointError, OverflowError, ArithmeticError):
        return None
    if np.isfinite(rgb).all() and np.all((rgb >= 0.) & (rgb <= MAXIMUM)):
        return rgb
    return None


def _cone_cap(context, ja, hue):
    """An outer ray bracket from the eight cube vertices in cone coordinates."""
    matrix = np.asarray(context.matrix, dtype=float)
    effective = np.column_stack((matrix[:, 0], 1-matrix[:, 0]-matrix[:, 2], matrix[:, 2]))
    inverse = np.linalg.inv(effective)
    low = MAXIMUM*np.minimum(inverse, 0.).sum(axis=1)
    high = MAXIMUM*np.maximum(inverse, 0.).sum(axis=1)

    def response(cone):
        p = (context.fl*abs(cone))**.42
        return np.sign(cone)*400*p/(27.13+p)

    response_low, response_high = response(low), response(high)
    neutral = context.aw/3.05*(ja/100.)**(1/GAMMA)
    direction = (context.aw/(1204*eccentricity(hue)))[..., None]*(
        np.cos(hue)[..., None]*INVERSE[:, 1]+np.sin(hue)[..., None]*INVERSE[:, 2])
    bounds = np.full(direction.shape, np.inf)
    positive = direction > 0
    negative = direction < 0
    neutral = np.asarray(neutral)[..., None]
    np.divide(response_high-neutral, direction, out=bounds, where=positive)
    np.divide(response_low-neutral, direction, out=bounds, where=negative)
    return np.min(bounds, axis=-1)


def _row(context, q, energy):
    white = context.white
    if energy <= 0:
        return np.zeros(3), 0., np.zeros(3)
    if energy >= white*white:
        return np.array([white, 0., 0.]), white*white, np.full(3, MAXIMUM)

    direct = _decoded(context, q, energy)
    if direct is not None:
        return q.copy(), energy, direct

    ja = np.sqrt(energy)
    chroma = np.hypot(q[1], q[2])
    if not chroma:
        raise ArithmeticError('An interior neutral must decode inside the cube')
    hue = np.arctan2(q[2], q[1])
    unit = q[1:]/chroma

    def candidate(radius):
        ab = unit*radius
        value = np.array([signed_hk(ja, radius), ab[0], ab[1]])
        return value, _decoded(context, value, energy)

    low, high = 0., min(chroma, _cone_cap(context, ja, hue))
    valid_q, valid_rgb = candidate(low)
    if valid_rgb is None:
        raise ArithmeticError('The neutral ray origin is outside the cube')
    if not np.isfinite(high) or high <= 0:
        raise ArithmeticError('The cube ray has no finite outer bracket')
    seed = float(context.capacity(ja, hue))
    if np.isfinite(seed) and 0 < seed < high:
        tested_q, tested_rgb = candidate(seed)
        if tested_rgb is None:
            high = seed
        else:
            low, valid_q, valid_rgb = seed, tested_q, tested_rgb

    tested_q, tested_rgb = candidate(high)
    if tested_rgb is not None:
        return tested_q, energy, tested_rgb
    for _ in range(FLOAT64_STEPS):
        middle = low+(high-low)/2
        if middle == low or middle == high:
            break
        tested_q, tested_rgb = candidate(middle)
        if tested_rgb is None:
            high = middle
        else:
            low, valid_q, valid_rgb = middle, tested_q, tested_rgb
    return valid_q, energy, valid_rgb


def _decode_rows(context, q, energy):
    """Test rows with identical arithmetic for batch and single-row callers."""
    rgb = np.full_like(q, np.nan)
    response = context.responses(q, energy)
    safe = np.isfinite(response).all(axis=-1) & np.all(abs(response) < 400., axis=-1)
    if np.any(safe):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            rgb[safe] = context.decode(q[safe], energy[safe])
    legal = np.isfinite(rgb).all(axis=-1) & np.all((rgb >= 0.) & (rgb <= MAXIMUM), axis=-1)
    return rgb, legal


def guard(context, q, energy):
    """Project appearance rows and return (coordinates, energy, tested RGB)."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (3,) or not np.isfinite(q).all():
        raise ValueError('Cube guard requires finite appearance triples')
    energy = np.broadcast_to(np.asarray(energy, dtype=float), q.shape[:-1])
    if not np.isfinite(energy).all():
        raise ValueError('Cube guard requires finite signed energy')
    rows = q.reshape(-1, 3)
    e = energy.ravel()
    out, stored, rgbs = rows.copy(), e.copy(), np.zeros_like(rows)
    black, white = e <= 0, e >= context.white**2
    out[black] = 0.; stored[black] = 0.
    out[white] = [context.white, 0., 0.]
    stored[white] = context.white**2; rgbs[white] = MAXIMUM
    ids = np.flatnonzero(~black & ~white)
    if not len(ids):
        return out.reshape(q.shape), stored.reshape(energy.shape), rgbs.reshape(q.shape)
    decoded, legal = _decode_rows(context, rows[ids], e[ids])
    rgbs[ids[legal]] = decoded[legal]
    ids = ids[~legal]
    if not len(ids):
        return out.reshape(q.shape), stored.reshape(energy.shape), rgbs.reshape(q.shape)
    source = rows[ids]; energies = e[ids]
    ja = np.sqrt(energies); c = np.hypot(source[:, 1], source[:, 2])
    if np.any(c == 0):
        raise ArithmeticError('An interior neutral must decode inside the cube')
    unit = source[:, 1:]/c[:, None]
    hue = np.arctan2(source[:, 2], source[:, 1])
    low = np.zeros_like(c); high = np.minimum(c, _cone_cap(context, ja, hue))
    if not np.isfinite(high).all() or np.any(high <= 0):
        raise ArithmeticError('The cube ray has no finite outer bracket')

    def candidate(radius):
        ab = unit*radius[:, None]
        trial = np.column_stack((signed_hk(ja, radius), ab))
        rgb, valid = _decode_rows(context, trial, energies)
        return trial, rgb, valid

    saved_q, saved_rgb, legal = candidate(low)
    if not np.all(legal):
        raise ArithmeticError('The neutral ray origin is outside the cube')
    seed = np.broadcast_to(np.asarray(context.capacity(ja, hue)), c.shape)
    use = np.isfinite(seed) & (seed > 0) & (seed < high)
    trial, rgb, legal = candidate(np.where(use, seed, low))
    accept = use & legal
    low[accept] = seed[accept]; saved_q[accept] = trial[accept]; saved_rgb[accept] = rgb[accept]
    high[use & ~legal] = seed[use & ~legal]
    trial, rgb, legal = candidate(high)
    low[legal] = high[legal]; saved_q[legal] = trial[legal]; saved_rgb[legal] = rgb[legal]
    for _ in range(FLOAT64_STEPS):
        middle = low+(high-low)/2
        moving = (middle != low) & (middle != high)
        if not np.any(moving):
            break
        trial, rgb, legal = candidate(middle)
        accept = moving & legal
        low[accept] = middle[accept]; saved_q[accept] = trial[accept]; saved_rgb[accept] = rgb[accept]
        high[moving & ~legal] = middle[moving & ~legal]
    out[ids] = saved_q; rgbs[ids] = saved_rgb
    return out.reshape(q.shape), stored.reshape(energy.shape), rgbs.reshape(q.shape)


def _decoded_scaled(context, q, energy):
    try:
        rgb = context.decode_scaled(q, energy)
    except (ValueError, FloatingPointError, OverflowError, ArithmeticError):
        return None
    return rgb if all(S() <= v <= MAXIMUM for v in rgb) else None


def guard_scaled(context, q, energy):
    """Scaled-number counterpart; retain tested RGB for the export adapter."""
    q, energy = list(map(S, q)), S(energy)
    white = S(context.white)
    if energy <= 0:
        return [S(), S(), S()], S(), [S(), S(), S()]
    if energy >= white*white:
        return [white, S(), S()], white*white, [S(MAXIMUM)]*3
    direct = _decoded_scaled(context, q, energy)
    if direct is not None:
        return q, energy, direct

    ja, chroma = energy**.5, norm(q[1:])
    if not chroma:
        raise ArithmeticError('An interior neutral must decode inside the cube')
    hue = np.arctan2(float(q[2]/chroma), float(q[1]/chroma))
    unit = [q[1]/chroma, q[2]/chroma]

    def candidate(radius):
        ab = [v*radius for v in unit]
        value = [signed_hk(ja, radius), *ab]
        return value, _decoded_scaled(context, value, energy)

    low = S()
    valid_q, valid_rgb = candidate(low)
    if valid_rgb is None:
        raise ArithmeticError('The neutral ray origin is outside the cube')
    t = (ja/white)**(1/GAMMA)
    # Physical capacity is rw*t*(1-t)*unit.  The stored fit remains a seed.
    from .cube_boundary import white_response
    seed = S(white_response(context.fl))*t*(1-t)*S(context.unit(float(t), hue))
    high = chroma
    if S() < seed < high:
        tested_q, tested_rgb = candidate(seed)
        if tested_rgb is None:
            high = seed
        else:
            low, valid_q, valid_rgb = seed, tested_q, tested_rgb
    tested_q, tested_rgb = candidate(high)
    if tested_rgb is not None:
        return tested_q, energy, tested_rgb
    for _ in range(FLOAT64_STEPS):
        if high <= low:
            break
        middle = (low*high)**.5 if low and high > 4*low else (low+high)/2
        if middle == low or middle == high:
            break
        tested_q, tested_rgb = candidate(middle)
        if tested_rgb is None:
            high = middle
        else:
            low, valid_q, valid_rgb = middle, tested_q, tested_rgb
    return valid_q, energy, valid_rgb
