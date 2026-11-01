"""Continuous signed tone equations and UI-coordinate adapters."""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from . import linear as cm

GAMMA = .525*(1.48+np.sqrt(.1))
# Derived by buck2 build //libraries/perceptual-grade-core:shadow_anchor.
LOW_Y = 0.015507796089266005
PEAKS = (500, 1000, 2000, 4000)
LEVEL_NAMES = ('Black', 'Shadows', 'Midtones', 'White', 'Highlights')
LEVEL_FIELDS = tuple('input_'+name.lower() for name in LEVEL_NAMES)
LEVEL_DEFAULTS = (0., LOW_Y*203., .18*203., 203., 1000.)
LEVEL_LIMITS = ((0., 2.), (.01, 50.), (.1, 250.), (10., 1000.), (100., 10000.))
PARAM_NAMES = (*LEVEL_FIELDS, 'fit', 'reference_nits', 'peak_nits')


@dataclass(frozen=True)
class Parameters:
    input_black: float = 0.
    input_shadows: float = LOW_Y*203.
    input_midtones: float = 36.54
    input_white: float = 203.
    input_highlights: float = 1000.
    fit: float = 0
    reference_nits: float = 203
    peak_nits: float = 1000


    @property
    def input_nits(self):
        return tuple(getattr(self, name) for name in LEVEL_FIELDS)

from ...color_core import modcam16 as responses
from ...color_core.modcam16 import viewing
from ...color_core.scaled import Scalar as S, dot, norm, smooth_floor, smooth_cap

def response(n, fl):
    p=fl**.42*np.abs(np.asarray(n,dtype=float))**.42
    return np.sign(n)*400*p/(27.13+p)


def unresponse(s, fl):
    a=np.abs(np.asarray(s,dtype=float))
    return np.sign(s)*(27.13*a/(400-a))**(1/.42)/fl


def saturation_gain(H, JA, C, H2):
    H,JA,C,H2=np.broadcast_arrays(H,JA,C,H2)
    denominator=np.hypot(66*C,2*JA*H2)+66*C
    return np.divide(2*H2*H2,denominator,out=np.zeros_like(H2,dtype=float),where=denominator>0)


def kernel(t, left, right, width):
    """Integral from 0 to t of sigmoid((v-l)/w)-sigmoid((v-r)/w).

For t>=0, K/w=log(1+(B-A)(U-1)/((A+1)(B+U))), where
A=exp(l/w), B=exp(r/w), U=exp(t/w). Evaluate the positive quotient in logs.
This remains accurate for narrow intervals and tiny t without quadrature.
"""
    if not (0 <= left < right and width > 0):
        raise ValueError('Kernel requires 0 <= left < right and width > 0')
    t = np.asarray(t, dtype=np.float64)
    gap = np.log(-np.expm1(-(right-left)/width))
    base = np.log1p(np.exp(-left/width))
    with np.errstate(divide='ignore', invalid='ignore'):
        positive = ((t-left)/width-base-np.logaddexp(0, (t-right)/width)
                    +gap+np.log(-np.expm1(-np.maximum(t, 0)/width)))
        negative = (-left/width-base-np.logaddexp(0, (t-right)/width)
                    +gap+np.log(-np.expm1(np.minimum(t, 0)/width)))
        out = np.where(t >= 0, width*np.logaddexp(0, positive),
                       width*np.log1p(-np.exp(negative)))
    return np.where(t == 0, 0., out)


def smoother(t):
    t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
    return t**3*(10+t*(-15+6*t))


def normalized_kernel(t, left, right, width):
    """K(t)/K(1) without overflowing or underflowing the width/normalizer.

    When the width is below the input spacing, retain its endpoint log(2)
    terms; exponentially small tails round to zero. At the opposite limit,
    the quotient is t up to O((max(1,|t|)/w)^2), below float64 rounding.
    """
    t = np.asarray(t, dtype=np.float64)
    width = S(width)
    if width > S(max(1., float(np.max(np.abs(t), initial=0))))*2.**27:
        return t.copy()
    if width < S(2.**-500):
        w = float(width)
        result = np.maximum(0., np.minimum(t, right)-left)
        result = np.where(t == left, w*np.log(2), result)
        result = np.where(t == right, right-left-w*np.log(2), result)
        base = right-left-(w*np.log(2) if left == 0 else 0.)
        return result/base
    w = float(width)
    return kernel(t, left, right, w)/kernel(1., left, right, w)


def fit_width(fit):
    return .05*(-S(fit)).exp2()


def normalized_scaled_kernel(t, left, right, width):
    t, width = S(t), S(width)
    if not t:
        return S()
    if t.e >= -500:
        return S(float(normalized_kernel(float(t), left, right, width)))
    if width > 2.**27:
        return t
    if width < S(2.**-500):
        if left > 0:
            return S()  # Tail is below rounding relative to the positive baseline.
        ratio = t/width
        if ratio > 2.**53:
            integral = t-width*np.log(2)
        elif ratio < S(2.**-52):
            integral = t/2
        else:
            v = float(ratio)
            integral = width*(np.log1p(np.expm1(v)/2) if v < 1 else np.logaddexp(0, v)-np.log(2))
        return integral/right
    w = float(width)
    a, b = np.exp(-left/w), np.exp(-right/w)
    slope = a*(-np.expm1(-(right-left)/w))/((1+a)*(1+b))
    return t*(slope/float(kernel(1., left, right, w)))


def scaled_weights(h, knots, high, fit, adaptance=False):
    t = min(S(1.), max(S(), S(h)/high))
    t = t**3*(10+t*(-15+6*t))
    width = fit_width(fit)
    u, previous = [], S(1.)
    for left, right in zip(knots[:-1], knots[1:]):
        previous = min(previous, max(S(), min(S(1.), normalized_scaled_kernel(t, left, right, width))))
        u.append(previous)
    return [(u[0] if adaptance else S(1.))-u[1], u[1]-u[2], u[2]-u[3], u[3]]


def coefficient_weights(h, knots, high, fit, adaptance=False):
    t = smoother(np.asarray(h)/high)
    width = fit_width(fit)
    u = np.stack([normalized_kernel(t, l, r, width)
                  for l, r in zip(knots[:-1], knots[1:])], axis=-1)
    u = np.minimum.accumulate(np.clip(u, 0, 1), axis=-1)
    first = u[..., 0] if adaptance else np.ones_like(t)
    return np.stack((first-u[..., 1], u[..., 1]-u[..., 2], u[..., 2]-u[..., 3], u[..., 3]), axis=-1)


class Model:
    def __init__(self, settings, space='Linear Rec.709'):
        self.params=settings; self.space=space
        self.settings = settings
        if not np.isfinite([*settings.input_nits, settings.fit, settings.reference_nits, settings.peak_nits]).all():
            raise ValueError('Tone parameters must be finite')
        self.reference = float(smooth_cap(smooth_floor(settings.reference_nits, 48.), 1000.))
        self.peak = float(smooth_cap(smooth_floor(settings.peak_nits, 2*self.reference), 10000.))
        self.target = self.nits_to_j([0., LOW_Y*self.reference, .18*self.reference, self.reference, self.peak])
        self.target[0], self.target[3] = 0., 1.
        self.source = self.nits_to_j(settings.input_nits)
        for i in range(1, 5):
            self.source[i] = max(self.source[i], self.source[i-1]+.001)
        self.black, self.span = self.source[0], self.source[-1]-self.source[0]
        self.knots = (self.source-self.black)/self.span
        self.width = fit_width(settings.fit)
        self.slope = np.min(np.diff(self.target)/np.diff(self.knots))
        self.residual = np.maximum(0, np.diff(self.target)-self.slope*np.diff(self.knots))
        self.active = not np.array_equal(self.source, self.target)
        self.high = 100*self.target[-1]
        self.coefficient_knots = smoother(self.target/self.target[-1])
        self.fl, self.aw = viewing(self.reference)
        self.w=np.asarray(cm.xyz(cm.white(space)))
        self.wc=np.asarray(cm.mv(cm.CAT16, self.w))
        self.rgb_to_xyz=np.asarray(cm.rgb_to_xyz(space))
        self.xyz_to_rgb=np.asarray(cm.xyz_to_rgb(space))
        self.to_cone=np.asarray(cm.CAT16)/self.wc[:,None] @ self.rgb_to_xyz
        self.from_cone=self.xyz_to_rgb@np.asarray(cm.ICAT16)@np.diag(self.wc)
        self.target_nits=np.array([0.,LOW_Y*self.reference,.18*self.reference,self.reference,self.peak])
        self.high_j=self.high
        self.plot_max=float(self.nits_to_j(4000))

    def nits_to_j(self, nits):
        fl, _ = viewing(self.reference)
        white_p = S(fl)**.42
        white_response = white_p/(27.13+white_p)
        result = []
        for value in np.asarray(nits, dtype=np.float64).flat:
            p = (fl*max(S(), S(value))/self.reference)**.42
            result.append(float(((p/(27.13+p))/white_response)**GAMMA))
        return np.asarray(result).reshape(np.shape(nits))

    def remap_h(self, h):
        h = np.asarray(h, dtype=np.float64)
        if not self.active:
            return h.copy()
        t = np.maximum(0, (h/100-self.black)/self.span)
        k = np.stack([normalized_kernel(t, l, r, self.width) for l, r in zip(self.knots[:-1], self.knots[1:])], axis=-1)
        out = 100*(self.slope*t+k @ self.residual)
        return np.where(t == 1, self.high, out)

    def apply_appearance(self, appearance):
        if not self.active:
            return appearance
        q, j = appearance.q, appearance.ja
        c = np.hypot(q[..., 1], q[..., 2])
        h = self.remap_h(np.hypot(j, np.sqrt(responses.HK*c)))
        denominator = np.hypot(responses.HK*c, 2*j*h)+responses.HK*c
        gain = np.divide(2*h*h, denominator, out=np.zeros_like(h), where=denominator > 0)
        ja, ab = gain*j, q[..., 1:]*gain[..., None]
        out = np.concatenate((responses.signed_hk(ja, np.hypot(ab[..., 0], ab[..., 1]))[..., None], ab), axis=-1)
        return responses.Appearance(out, ja)

    def scaled_apply(self, q, ja):
        if not self.active:
            return q, ja
        c = norm(q[1:])
        magnitude = norm([ja, (responses.HK*c)**.5])
        t = max(S(), (magnitude/100-self.black)/self.span)
        if t.e >= -500 or not t:
            h = S(float(self.remap_h(float(magnitude))))
        else:
            values = [normalized_scaled_kernel(t, left, right, self.width)
                      for left, right in zip(self.knots[:-1], self.knots[1:])]
            h = 100*(self.slope*t+dot(values, self.residual))
        denominator = norm([responses.HK*c, 2*ja*h])+responses.HK*c
        gain = 2*h*h/denominator if denominator else S()
        ja, ab = ja*gain, [v*gain for v in q[1:]]
        return [responses.signed_hk(ja, norm(ab)), *ab], ja

    def remap(self, j):
        return self.remap_h(100*np.asarray(j))/100

    def j_to_nits(self, j):
        j=np.asarray(j,dtype=float)
        value=self.aw*np.maximum(j,0)**(1/GAMMA)/3.05
        return unresponse(value,self.fl)*self.reference

    def cones(self, rgb):
        rgb=np.asarray(rgb,dtype=float)
        change=(rgb-rgb[...,1,None])@self.rgb_to_xyz.T
        return rgb[...,1,None]+change @ (np.asarray(cm.CAT16)/self.wc[:,None]).T

    def appearance(self, rgb):
        return responses.encode_cones(self.cones(rgb),self.reference).q

    def stage2(self, rgb):
        return self.apply_appearance(responses.encode_cones(self.cones(rgb),self.reference))

    def apply(self, rgb):
        from .forward import Model
        return Model(self).apply(rgb)


@lru_cache(maxsize=64)
def model(params,space='Linear Rec.709'):
    return Model(params,space)


def clamp_landmark(params,index,value,space='Linear Rec.709'):
    """Clamp the edited source landmark, never push a neighboring control."""
    m=model(params,space); lo,hi=LEVEL_LIMITS[index]
    j=float(m.nits_to_j(max(lo,min(hi,value))))
    low=float(m.nits_to_j(lo)); high=float(m.nits_to_j(hi))
    if index: low=max(low,float(m.nits_to_j(getattr(params,LEVEL_FIELDS[index-1])))+.001)
    if index<4: high=min(high,float(m.nits_to_j(getattr(params,LEVEL_FIELDS[index+1])))-.001)
    return float(m.j_to_nits(max(low,min(high,j))))
