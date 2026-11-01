"""Reference export-only H–K repair, before the independently validated cube guard.

The post-tone reference energy is fixed during each forward evaluation. Raw
signed handoffs are never repaired. The optional standalone inverse is a math
probe; the RGB inverse instead evaluates this closed-form forward in its residual.
"""
from dataclasses import dataclass
import numpy as np
from ...color_core.modcam16 import State, HK
from ...color_core.scaled import Scalar as S, norm

LIFT = .20
ALLOCATION = .44  # (1 + LIFT)^2 - 1, exactly 11/25 in the proofs.


@dataclass(frozen=True)
class GradedState(State):
    """Raw state plus export context; q/energy keep the ordinary State meaning."""
    reference_energy: np.ndarray


def harmonic(a, b):
    low, high = np.minimum(a, b), np.maximum(a, b)
    ratio = np.divide(low, high, out=np.zeros_like(low), where=high > 0)
    return low/(1+ratio)


def requested_square(energy, chroma):
    """H²−66C from carried signed energy, including the reflected H branch.

    E = u(|u|+2b), u=H-b, b=sqrt(66C). Thus D=E for E>=0,
    and D=E+2u² otherwise. Recover u by rationalization, preserving tiny
    achromatic departures that the rounded H coordinate cannot represent.
    """
    e, c = np.broadcast_arrays(np.asarray(energy, float), np.asarray(chroma, float))
    b = np.sqrt(HK*c)
    denominator = np.hypot(np.sqrt(abs(e)), b)+b
    u = np.divide(e, denominator, out=np.zeros_like(e), where=denominator > 0)
    return np.where(e < 0, e+2*u*u, e)


def guard_scaled(q, energy, reference_energy):
    """Same equations using integer exponents for finite extreme controls."""
    q = list(map(S, q)); e = S(energy); e0 = max(S(), S(reference_energy))
    h = q[0]
    if h < 0:
        return q, e
    c = norm(q[1:]); b = (HK*c)**.5
    denominator = norm([abs(e)**.5, b])+b
    u = e/denominator if denominator else S()
    d = e+2*u*u if e < 0 else e
    x = h*h
    low, high = min(e0, x), max(e0, x)
    delta = low/(1+low/high) if high else S()
    if d >= delta:
        return q, e
    if delta:
        v = delta/(delta+max(delta-d, S()))
        w = 1-v
        repaired = delta*v**.5/(v*v+4*v*w+10*w*w)**.25
    else:
        repaired = S()
    z = max(S(), repaired-d); l = ALLOCATION*x
    den = norm([2*l, z])
    g = l*(z/den) if den else S()
    cg = min(c, max(S(), (x+g-repaired)/HK))
    gain = cg/c if c else S(1)
    return [(x+g)**.5, q[1]*gain, q[2]*gain], repaired


def guard(q, energy, reference_energy):
    """Return (repaired coordinates, carried nonnegative energy) on H>=0.

    Negative H retains the raw signed state for the cube guard's black policy.
    Nonfinite inputs are rejected. Reference energy is clamped at zero; a
    zero-knee negative deficit maps to zero energy and is not invertible.
    """
    q = np.asarray(q, float)
    if q.shape[-1:] != (3,) or not np.isfinite(q).all():
        raise ValueError('H-K repair requires finite appearance triples')
    e = np.broadcast_to(np.asarray(energy, float), q.shape[:-1])
    e0 = np.broadcast_to(np.asarray(reference_energy, float), e.shape)
    if not (np.isfinite(e).all() and np.isfinite(e0).all()):
        raise ValueError('H-K repair requires finite carried and reference energies')
    # Keep the vector intermediates within binary64. Extreme rows use the
    # same equations with integer exponents; no control limit is introduced.
    maximum = max(np.max(abs(q), initial=0), np.max(abs(e), initial=0), np.max(abs(e0), initial=0))
    tiny = np.any((abs(e) > 0) & (abs(e) < 2.**-400)) or np.any((abs(e0) > 0) & (abs(e0) < 2.**-400))
    if maximum > 2.**400 or tiny:
        pairs = [guard_scaled(row, value, ref) for row, value, ref in
                 zip(q.reshape(-1, 3), e.ravel(), e0.ravel())]
        return (np.asarray([[float(v) for v in row] for row, _ in pairs]).reshape(q.shape),
                np.asarray([float(value) for _, value in pairs]).reshape(e.shape))
    out, stored = q.copy(), e.copy()
    rows, values, references = out.reshape(-1, 3), stored.reshape(-1), e0.reshape(-1)
    c = np.hypot(rows[:, 1], rows[:, 2]); x = rows[:, 0]**2
    d = requested_square(values, c)
    delta = harmonic(np.maximum(references, 0), x)
    ids = np.flatnonzero((rows[:, 0] >= 0) & (d < delta))
    if not len(ids):
        return out, stored
    dd, xx, cc, knee = d[ids], x[ids], c[ids], delta[ids]
    v = np.divide(knee, knee+np.maximum(knee-dd, 0), out=np.zeros_like(knee), where=knee > 0)
    w = 1-v
    repaired = knee*np.sqrt(v)/np.sqrt(np.sqrt(v*v+4*v*w+10*w*w))
    z = np.maximum(repaired-dd, 0); l = ALLOCATION*xx
    den = np.hypot(2*l, z)
    g = l*np.divide(z, den, out=np.zeros_like(z), where=den > 0)
    cg = np.minimum(cc, np.maximum(0, (xx+g-repaired)/HK))
    gain = np.divide(cg, cc, out=np.ones_like(cc), where=cc > 0)
    rows[ids, 0] = np.sqrt(xx+g)
    rows[ids, 1:] *= gain[:, None]
    values[ids] = repaired
    return out, stored


@dataclass(frozen=True)
class GuardInverseResult:
    state: State
    residual: np.ndarray
    status: np.ndarray


def inverse(q, energy, reference_energy):
    """Fixed-reference appearance probe, never called by the RGB solver.

    Positive E/E0 has a unique scalar inverse. A bracketed binary64 solve
    recovers it; every result includes a measured forward residual. Zero E
    or E0 is reported as degenerate, not as a unique round trip. Invalid
    redundant H/E/C records are rejected rather than silently reinterpreted.
    """
    q = np.asarray(q, float)
    if q.shape[-1:] != (3,) or not np.isfinite(q).all():
        raise ValueError('H-K inverse requires finite appearance triples')
    e = np.broadcast_to(np.asarray(energy, float), q.shape[:-1])
    e0 = np.broadcast_to(np.asarray(reference_energy, float), e.shape)
    c = np.hypot(q[..., 1], q[..., 2]); u = e+HK*c
    if (not (np.isfinite(e).all() and np.isfinite(e0).all() and np.isfinite(u).all())
            or np.any(e < 0) or np.any(e0 < 0) or np.any(q[..., 0] < 0)
            or np.any(abs(q[..., 0]-np.sqrt(u)) > 2e-12*(1+np.sqrt(u)))):
        raise ValueError('Inconsistent or nonphysical guarded appearance')
    out = q.copy(); raw_e = e.copy()
    statuses = np.full(e.shape, 'identity', dtype='<U11')
    degenerate = (e == 0) | (e0 == 0)
    statuses[degenerate] = 'degenerate'
    active = (~degenerate) & (e < harmonic(e0, u))
    rows = out.reshape(-1, 3)
    for i in np.flatnonzero(active.ravel()):
        ei, er, ui = e.ravel()[i], e0.ravel()[i], u.ravel()[i]
        # Normalize by U. The analytic bracket follows G<=.44X and delta>=E.
        lo = max(1/(1+ALLOCATION), (ei/ui)/(1-ei/er)); hi = 1.
        def quantities(x):
            X = S(x)*ui
            delta = min(S(er), X)/(1+min(S(er), X)/max(S(er), X))
            ratio = max(S(1), delta/ei)
            # Positive-root formula rationalized near ratio=1.
            r = float(ratio) if ratio < 2 else None
            if r is not None:
                rm1 = np.expm1(4*np.log(r))
                t = S(rm1/(np.sqrt(4+10*rm1)+2))
            else:
                t = (ratio*ratio*(10-6/ratio**4)**.5-2)/10
            z = max(S(), S(ei)-delta+delta*t)
            l = ALLOCATION*X
            g = l*z/norm([2*l, z])
            return float((X+g)/ui)-1, X, z
        for _ in range(56):
            middle = lo+(hi-lo)/2
            if middle == lo or middle == hi:
                break
            f, _, _ = quantities(middle)
            if f > 0: hi = middle
            else: lo = middle
        _, x, z = quantities(lo+(hi-lo)/2)
        raw_c = float((x-S(ei)+z)/HK)
        rows[i, 0] = float(x**.5)
        # Normalize first: C_raw/C_guarded can overflow for a deep repair
        # even when each radius and both reconstructed opponents are finite.
        rows[i, 1:] = (rows[i, 1:]/c.ravel()[i])*raw_c
        # The scalar solve also recovers D. Do not lose this information by
        # subtracting reconstructed H and sqrt(66C), which may round equal.
        d = S(ei)-z
        b = (x-d)**.5
        offset = d/(x**.5+b) if x or b else S()
        raw_e.reshape(-1)[i] = float(d-2*offset*offset if d < 0 else d)
        statuses.reshape(-1)[i] = 'root'
    checked, ce = guard(out, raw_e, e0)
    errors = np.maximum(np.max(abs(checked-q), axis=-1)/(1+np.max(abs(q), axis=-1)), abs(ce-e)/(1+abs(e)))
    statuses = np.where(active & (errors > 2e-10), 'approximate', statuses)
    return GuardInverseResult(State(out, raw_e), errors, statuses)
