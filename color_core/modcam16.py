"""Grade's signed, offset-free modCAM16 response equations and HK functions.

This variant preserves signed achromatic energy and explicit decoder domains.
"""
from dataclasses import dataclass
import numpy as np
from .scaled import Scalar as S, dot, norm, signed_power, total

B = 400.
POWER = .42
HK = 66.
GAMMA = .525*(1.48+np.sqrt(.1))
OPPONENT = np.array([[1., -12/11, 1/11], [1/9, 1/9, -2/9]])
INVERSE = np.array([[460., 451., 288.], [460., -891., -261.], [460., -220., -6300.]])/1403


@dataclass(frozen=True)
class Response:
    value: S
    margin: S  # B - abs(value), including when value rounds to +/-B.


def compress(cone, fl):
    cone = S(cone)
    p = (fl*abs(cone))**POWER
    return Response((1 if cone >= 0 else -1)*B*p/(27.13+p), B*27.13/(27.13+p))


def uncompress(response, fl):
    if response.margin <= 0:
        raise ValueError('Uncompression requires a strictly positive pole margin')
    return (1 if response.value >= 0 else -1)*(27.13*abs(response.value)/response.margin)**(1/POWER)/fl


def eccentricity(angle):
    h = np.deg2rad(np.rad2deg(angle) % 360)
    return (1-.0582*np.cos(h)-.0258*np.cos(2*h)-.1347*np.cos(3*h)+.0289*np.cos(4*h)
            -.1475*np.sin(h)-.0308*np.sin(2*h)+.0385*np.sin(3*h)+.0096*np.sin(4*h))


def signed_hk(ja, chroma):
    """Reflect H about sqrt(66C) on the negative achromatic side."""
    if isinstance(ja, S):
        b = (HK*chroma)**.5
        magnitude = norm([ja, b])
        return magnitude if ja >= 0 else 2*b-magnitude
    b = np.sqrt(HK*np.asarray(chroma))
    magnitude = np.hypot(ja, b)
    return np.where(np.asarray(ja) >= 0, magnitude, 2*b-magnitude)


def hk_energy(h, chroma):
    """Continuous signed inverse E=JA*abs(JA); all real H are defined."""
    b = (HK*chroma)**.5 if isinstance(h, S) else np.sqrt(HK*np.asarray(chroma))
    u = h-b
    return u*(abs(u)+2*b) if isinstance(h, S) else u*(np.abs(u)+2*b)


def achromatic_response(energy, aw):
    """Authoritative requested A from carried signed energy, without cancellation."""
    return aw*signed_power(energy, 1/(2*GAMMA))/100**(1/GAMMA)


def requested_energy(q, base_q, base_ja):
    """Carry the anchor energy through signed coordinate changes.

    u=E/(M+B) retains tiny signed departures that rounded H cannot store.
    The difference of u*abs(u) is factored; near reset E+delta is exact at
    zero change, while the direct form avoids large-change cancellation.
    """
    if isinstance(base_ja, S):
        c, c0 = norm(q[1:]), norm(base_q[1:])
        b, b0 = (HK*c)**.5, (HK*c0)**.5
        e0 = base_ja*abs(base_ja)
        denominator = norm([base_ja, b0])+b0
        u0 = e0/denominator if denominator else S()
        db = HK*(c-c0)/(b+b0) if b+b0 else S()
        du = q[0]-base_q[0]-db
        u = u0+du
        r = abs(u)+abs(u0)
        slope = r if u*u0 >= 0 else (u*u+u0*u0)/r
        change = total([du*(slope+2*b), 2*u0*db])
        return e0+change if abs(change) <= .25*abs(e0) else u*(abs(u)+2*b)
    c, c0 = np.hypot(q[..., 1], q[..., 2]), np.hypot(base_q[..., 1], base_q[..., 2])
    b, b0 = np.sqrt(HK*c), np.sqrt(HK*c0)
    e0 = base_ja*np.abs(base_ja)
    denominator = np.hypot(base_ja, b0)+b0
    u0 = np.divide(e0, denominator, out=np.zeros_like(e0), where=denominator > 0)
    db = np.divide(HK*(c-c0), b+b0, out=np.zeros_like(c), where=b+b0 > 0)
    du = q[..., 0]-base_q[..., 0]-db
    u = u0+du
    r = np.abs(u)+np.abs(u0)
    opposite = np.divide(u*u+u0*u0, r, out=np.zeros_like(r), where=r > 0)
    slope = np.where(u*u0 >= 0, r, opposite)
    change = du*(slope+2*b)+2*u0*db
    return np.where(np.abs(change) <= .25*np.abs(e0), e0+change, u*(np.abs(u)+2*b))


def coordinates(responses, aw):
    values = [r.value for r in responses]
    ac = dot([2., 1., .05], values)
    # Opponent rows sum to zero. Margins retain chroma near the white pole;
    # values retain it near black (without adding and subtracting Colour's .1).
    same_sign = all(v >= 0 for v in values) or all(v <= 0 for v in values)
    near_pole = same_sign and max(abs(v) for v in values) > B/2
    sign = 1 if ac >= 0 else -1
    ab = [(-sign*dot(row, [r.margin for r in responses]) if near_pole else dot(row, values))
          for row in OPPONENT]
    radius = norm(ab)
    angle = np.arctan2(float(ab[1]/radius), float(ab[0]/radius)) if radius else 0.
    factor = 1204*eccentricity(angle)/aw
    ab = [v*factor for v in ab]
    ja = 100*signed_power(ac/aw, GAMMA)
    return [signed_hk(ja, norm(ab)), *ab], ja


@dataclass(frozen=True)
class Appearance:
    q: np.ndarray
    ja: np.ndarray


@dataclass(frozen=True)
class State:
    """Appearance handoff; signed energy remains authoritative at H-K cancellation."""
    q: np.ndarray
    energy: np.ndarray

    @property
    def ja(self):
        e=np.asarray(self.energy)
        if e.dtype==object:
            return np.asarray([signed_power(v,.5) for v in e.flat],dtype=object).reshape(e.shape)
        return np.sign(e)*np.sqrt(abs(e))



def viewing(reference):
    la=.1*reference; k=1/(5*la+1)
    fl=.2*k**4*5*la+.1*(1-k**4)**2*(5*la)**(1/3)
    p=fl**.42
    return fl, 3.05*400*p/(27.13+p)


def encode_cones(cones, reference=203., constants=None):
    """Offset-free algebraic form of Colour's adapted-cone appearance components.

    Signed coordinates avoid an XYZ round trip and retain achromatic sign.
    The external XYZ_to_Hellwig2022 equations are unchanged; removing the .1
    offset keeps subnormal responses distinct from camera black.
    """
    cones = np.asarray(cones, dtype=np.float64)
    if cones.shape[-1:] != (3,) or not np.isfinite(cones).all():
        raise ValueError('Cones must be finite triples')
    fl, aw = viewing(reference) if constants is None else constants
    p = fl**.42*np.abs(cones)**.42
    responses = np.sign(cones)*400*p/(27.13+p)
    margins = 400*27.13/(27.13+p)
    ac = responses @ np.array([2., 1., .05])
    ab = responses @ OPPONENT.T
    same_sign = np.all(cones >= 0, axis=-1) | np.all(cones <= 0, axis=-1)
    near_pole = same_sign & (np.max(np.abs(responses), axis=-1, initial=0) > 200)
    sign = np.where(ac >= 0, 1., -1.)
    ab = np.where(near_pole[..., None], -sign[..., None]*(margins @ OPPONENT.T), ab)
    angle = np.arctan2(ab[..., 1], ab[..., 0])
    ab *= (1204*eccentricity(angle)/aw)[..., None]
    neutral = np.all(cones == cones[..., :1], axis=-1)
    ab = np.where(neutral[..., None], 0., ab)
    ja = 100*signed_power(ac/aw, GAMMA)
    c = np.hypot(ab[..., 0], ab[..., 1])
    return Appearance(np.concatenate((signed_hk(ja, c)[..., None], ab), axis=-1), ja)
