"""Linear and bounded appearance-saturation charts shared by reference and Blender.

Coordinates retain opponent order (a, b); the UI alone rotates them to (-b, a).
Energy is carried signed J_A squared, never reconstructed from rounded J_HK.
"""
import numpy as np
from ...color_core.scaled import Scalar as S, norm


def project(q, energy):
    """Return (a,b)/sqrt(abs(E)), with NaN for chromatic E=0 states.

    Neutrals, including black and negative neutrals, have the directionless
    center value. Negative nonzero energy has the same finite chart scale as
    positive energy; callers retain the energy sign for branch styling.
    """
    q = np.asarray(q)
    if q.shape[-1:] != (3,):
        raise ValueError('Saturation projection requires appearance triples')
    energy = np.broadcast_to(np.asarray(energy), q.shape[:-1])
    if q.dtype == object or energy.dtype == object:
        rows = []
        for value, stored in zip(q.reshape(-1, 3), energy.reshape(-1)):
            try:
                ab = [S(v) for v in value[1:]]
                e = S(stored)
            except ValueError:
                rows.append([np.nan, np.nan])
                continue
            if not any(ab):
                rows.append([0., 0.])
            elif not e:
                rows.append([np.nan, np.nan])
            else:
                denominator = abs(e)**.5
                try:
                    rows.append([float(v/denominator) for v in ab])
                except OverflowError:
                    rows.append([np.nan, np.nan])
        return np.asarray(rows, float).reshape(q.shape[:-1]+(2,))
    ab = np.asarray(q[..., 1:], float)
    energy = np.asarray(energy, float)
    denominator = np.sqrt(abs(energy))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.divide(ab, denominator[..., None], out=np.full_like(ab, np.nan),
                           where=(denominator > 0)[..., None])
    neutral = np.all(ab == 0, axis=-1) & np.isfinite(energy)
    result = np.where(np.isfinite(energy)[..., None], result, np.nan)
    return np.where(neutral[..., None], 0., result)


def valid(q, energy):
    """Whether a state has finite coordinates on the linear saturation pad."""
    return np.isfinite(project(q, energy)).all(axis=-1)


def bounded_project(q, energy):
    """Compact signed-energy appearance saturation into the closed unit disk.

    The origin represents a neutral regardless of energy. Chromatic zero-energy
    states land on the unit circle. Energy's sign is carried separately by the
    caller; both signs share the same chart magnitude.
    """
    q = np.asarray(q)
    if q.shape[-1:] != (3,):
        raise ValueError('Bounded saturation projection requires appearance triples')
    energy = np.broadcast_to(np.asarray(energy), q.shape[:-1])
    if q.dtype == object or energy.dtype == object:
        rows = []
        for value, stored in zip(q.reshape(-1, 3), energy.reshape(-1)):
            try:
                ab = [S(v) for v in value[1:]]
                e = S(stored)
                if not any(ab):
                    rows.append([0., 0.])
                    continue
                # Retain exponents through the norm. Conversion to float is
                # safe only after each quotient has magnitude at most one.
                denominator = norm([*ab, abs(e)**.5])
                rows.append([float(v/denominator) for v in ab])
            except (ValueError, OverflowError, ZeroDivisionError):
                rows.append([np.nan, np.nan])
        return np.asarray(rows, float).reshape(q.shape[:-1]+(2,))
    ab = np.asarray(q[..., 1:], float)
    energy = np.asarray(energy, float)
    root_energy = np.sqrt(abs(energy))
    scale = np.maximum(np.max(abs(ab), axis=-1), root_energy)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        normalized = ab/scale[..., None]
        denominator = np.hypot(np.hypot(normalized[..., 0], normalized[..., 1]), root_energy/scale)
        result = normalized/denominator[..., None]
    neutral = np.all(ab == 0, axis=-1) & np.isfinite(energy)
    result = np.where(np.isfinite(energy)[..., None], result, np.nan)
    return np.where(neutral[..., None], 0., result)


def bounded_from_linear(p):
    """Compact finite linear saturation coordinates without squaring them."""
    p = np.asarray(p, float)
    if p.shape[-1:] != (2,):
        raise ValueError('Linear saturation coordinates must be pairs')
    return bounded_project(np.concatenate((np.zeros(p.shape[:-1]+(1,)), p), axis=-1), 1.)


def bounded_unproject(p):
    """Release an open-disk coordinate to the existing linear target solver.

    The unit circle has infinite linear saturation and cannot be a finite
    draggable destination for a corner on a fixed nonzero-energy branch.
    """
    p = np.asarray(p, float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('Bounded saturation target must be a finite pair')
    radius = np.hypot(*p)
    if radius >= 1.:
        raise ValueError('Bounded saturation target must lie inside the unit circle')
    denominator = np.sqrt((1.-radius)*(1.+radius))
    return p/denominator
