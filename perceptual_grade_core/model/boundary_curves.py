"""C2 cached curves, evaluated independently of the reference implementation."""
import numpy as np

FRACTION = .25


def stencil(knots, x, period=0.):
    knots, x = np.asarray(knots, dtype=float), np.asarray(x, dtype=float)
    count = len(knots)
    x = x % period if period else np.clip(x, knots[0], knots[-1])
    i = np.searchsorted(knots, x, side='right')-1
    if not period:i = np.clip(i, 0, count-2)
    j = (i+1) % count
    a, b = knots[i], np.where(j == 0, period, knots[j])
    t = (x-a)/(b-a)
    indices = np.stack((i, j, j), axis=-1)
    weights = np.stack((1-t, t, np.zeros_like(t)), axis=-1)
    for k, center in ((i, a), (j, b)):
        if period:
            dl = (knots[k]-knots[(k-1) % count]) % period
            dr = (knots[(k+1) % count]-knots[k]) % period
        else:
            dl = knots[k]-knots[np.maximum(0, k-1)]
            dr = knots[np.minimum(count-1, k+1)]-knots[k]
        r = FRACTION*np.minimum(dl, dr)
        active = (r > 0) & (abs(x-center) < r)
        safe_r = np.where(active, r, 1.)
        z = x-center
        excess = np.maximum(0., safe_r-abs(z))
        positive = np.maximum(z, 0.)+excess**3/(6*safe_r*safe_r)
        wl = (positive-z)/np.where(active, dl, 1.)
        wr = positive/np.where(active, dr, 1.)
        indices = np.where(active[..., None], np.stack(((k-1) % count, k, (k+1) % count), axis=-1), indices)
        weights = np.where(active[..., None], np.stack((wl, 1-wl-wr, wr), axis=-1), weights)
    return indices, weights


def root(a, b, value):
    linear = value/a
    argument = np.sqrt(27*abs(b)/a)*linear/2
    angle = np.where(b < 0, np.arcsin(np.minimum(1., argument)), np.arcsinh(argument))/3
    wave = np.where(b < 0, np.sin(angle), np.sinh(angle))
    return linear*np.divide(3*wave, argument, out=np.ones_like(linear), where=argument != 0)


