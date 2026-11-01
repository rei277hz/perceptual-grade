"""Sample population and spatial coordinates, separate from source opacity."""
import numpy as np


def values(grid, shape):
    """Validate value-only metadata; legacy rectangular sources have unit mass."""
    if grid is None:
        mass = np.ones(shape, dtype=np.float64)
        x = np.broadcast_to(np.arange(shape[1], dtype=np.float64)[None, :] / shape[1], shape)
        return mass, x
    mass = np.asarray(grid['mass'], dtype=np.float64)
    x = np.asarray(grid['x'], dtype=np.float64)
    if mass.shape != shape or x.shape != shape:
        raise ValueError('Scope sample mass and coordinates must match the RGBA grid')
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Scope sample mass must be finite and nonnegative')
    if not np.isfinite(x).all() or np.any((mass > 0) & ((x < 0) | (x >= 1))):
        raise ValueError('Scope sample X must be normalized within the source atlas')
    return mass, x


def sample(points, opacity, mass, keys=None):
    """Keep legacy hash selection; resample nonuniform populations before fitting."""
    from . import statistics
    mass = np.asarray(mass).reshape(-1)
    positive = mass[mass > 0]
    weights = np.asarray(opacity).reshape(-1) * mass
    if len(positive) and np.any(positive != positive[0]):
        return statistics.weighted_sample(points, weights)
    return statistics.hashed_sample(points, weights, keys=keys)


def key(grid):
    return None if grid is None else grid.get('key')
