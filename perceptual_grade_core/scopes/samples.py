"""Deterministic scope sampling coordinates, independent of Blender and pixels.

Mass describes the pixel population represented by a sample. It is independent
of alpha/coverage. Reference inputs reuse the primary layout, including mass.
"""
from dataclasses import dataclass
import math
import numpy as np


MAX_SAMPLES = 256 * 192


def immutable(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class SampleLayout:
    mass: np.ndarray
    x: np.ndarray
    uv: np.ndarray
    tiles: np.ndarray
    ids: np.ndarray
    key: tuple
    tiled: bool = False

    @property
    def shape(self):
        return self.mass.shape

    @property
    def count(self):
        return int(np.count_nonzero(self.mass))


def sample_size(width, height, maximum=(256, 192)):
    if width < 1 or height < 1:
        raise ValueError('The scope image has no loaded pixels')
    scale = min(1., maximum[0] / width, maximum[1] / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def ordinary_layout(width, height, maximum=(256, 192)):
    width, height = int(width), int(height)
    sw, sh = sample_size(width, height, maximum)
    us=(np.floor((np.arange(sw)+.5)*width/sw)+.5)/width
    vs=(np.floor((np.arange(sh)+.5)*height/sh)+.5)/height
    u,v=np.meshgrid(us,vs)
    return SampleLayout(immutable(np.ones((sh, sw))), immutable(u),
                        immutable(np.stack((u, v), axis=-1)),
                        immutable(np.full((sh, sw), 1001), np.int32),
                        immutable(np.arange(sw * sh).reshape(sh, sw), np.int64),
                        ('scope-grid-v1', width, height, sw, sh))


def tile_allocations(tiles, maximum=MAX_SAMPLES):
    """Reserve one sample per tile, then use exact largest-remainder quotas."""
    tiles = tuple(sorted((int(t), int(w), int(h)) for t, w, h in tiles))
    if not tiles or len({t for t, _, _ in tiles}) != len(tiles):
        raise ValueError('Scope tiles must be a nonempty set of unique tile numbers')
    if any(t < 1001 or min(w, h) < 1 for t, w, h in tiles):
        raise ValueError('Scope tiles require valid numbers and nonempty pixels')
    total = sum(w * h for _, w, h in tiles)
    count = min(total, int(maximum))
    if len(tiles) > count:
        raise ValueError('The scope sample budget cannot represent every tile')
    remainder = count - len(tiles)
    capacity = total - len(tiles)
    numerators = [remainder * (w * h - 1) for _, w, h in tiles]
    allocations = [1 + (value // capacity if capacity else 0) for value in numerators]
    extra = count - sum(allocations)
    order = sorted(range(len(tiles)),
                   key=lambda i: (-(numerators[i] % capacity if capacity else 0), tiles[i][0]))
    for i in order[:extra]:
        allocations[i] += 1
    return tiles, tuple(allocations), total, count


def tiled_layout(tiles, maximum=MAX_SAMPLES):
    """Sample tiles by pixel population; retain atlas X and local UV separately."""
    tiles, allocations, total, count = tile_allocations(tiles, maximum)
    width = min(256, count)
    height = (count + width - 1) // width
    mass = np.zeros(height * width)
    uv = np.zeros((height * width, 2))
    atlas_x = np.zeros(height * width)
    tile_ids = np.zeros(height * width, np.int32)
    ids = np.full(height * width, -1, np.int64)
    columns = [(t - 1001) % 10 for t, _, _ in tiles]
    first, span = min(columns), max(columns) - min(columns) + 1
    cursor = 0
    for (tile, w, h), n in zip(tiles, allocations):
        rows = int(math.floor(math.sqrt(n * h / w) + .5))
        rows = max((n + w - 1) // w, min(h, n, rows))
        base, extras = divmod(n, rows)
        used = 0
        for row in range(rows):
            cells = base + (row < extras)
            segment = slice(cursor + used, cursor + used + cells)
            us = (np.floor((np.arange(cells)+.5)*w/cells)+.5)/w
            # Row height is proportional to its population; cells have equal area.
            vs = (math.floor((used+cells*.5)*h/n)+.5)/h
            uv[segment, 0], uv[segment, 1] = us, vs
            atlas_x[segment] = (((tile - 1001) % 10 - first) + us) / span
            used += cells
        segment = slice(cursor, cursor + n)
        mass[segment] = (w * h / n) * (count / total)
        tile_ids[segment] = tile
        ids[segment] = np.arange(cursor, cursor + n)
        cursor += n
    shape = height, width
    return SampleLayout(immutable(mass.reshape(shape)), immutable(atlas_x.reshape(shape)),
                        immutable(uv.reshape((*shape, 2))), immutable(tile_ids.reshape(shape)),
                        immutable(ids.reshape(shape)), ('scope-tiles-v1', tiles, allocations), True)


def pixel_indices(layout, width, height):
    """Nearest source pixel at each cell midpoint, preserving bottom-up rows."""
    if min(width, height) < 1:
        raise ValueError('The scope image has no loaded pixels')
    x = np.minimum(width - 1, (layout.uv[..., 0] * width).astype(np.int64))
    y = np.minimum(height - 1, (layout.uv[..., 1] * height).astype(np.int64))
    return x, y


def sample_rgba(pixels, layout):
    pixels = np.asarray(pixels)
    height, width = pixels.shape[:2]
    x, y = pixel_indices(layout, width, height)
    result = np.array(pixels[y, x], copy=True)
    result[layout.mass == 0] = 0
    return immutable(result)
