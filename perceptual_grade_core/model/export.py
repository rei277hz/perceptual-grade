"""Native AP1 input projection; forward export uses the decoded cube guard."""
import numpy as np

MAXIMUM=1024.

def down(x):return np.nextafter(x,-np.inf)


def input_repair(rgb, weights):
    """Continuous idempotent input projection; identity throughout the LUT cube."""
    rgb = np.asarray(rgb, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if rgb.shape[-1:] != (3,) or not np.isfinite(rgb).all():
        raise ValueError('LUT input must contain finite RGB triples')
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('AP1 luminance weights must be positive')
    result = rgb.copy()
    active = (rgb.min(axis=-1) < 0) | (rgb.max(axis=-1) > MAXIMUM)
    x = rgb[active]
    if not len(x):return result
    amplitude = np.max(abs(x), axis=-1)
    unit = x/amplitude[:, None]
    y = np.maximum(0., unit @ weights)
    k = np.maximum(0., -unit.min(axis=-1))
    factor = np.divide(y, y+k, out=np.zeros_like(y), where=y+k > 0)
    v = np.where((k > 0)[:, None], (unit+k[:, None])*factor[:, None], unit)
    maximum = np.max(v, axis=-1)
    with np.errstate(over='ignore', divide='ignore'):
        scale = np.minimum(amplitude, down(np.divide(MAXIMUM, maximum, out=np.zeros_like(maximum), where=maximum > 0)))
    result[active] = v*np.maximum(0.,scale)[:, None]
    return result
