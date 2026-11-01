"""Portable LUT resource, kernel, and payload preparation helpers.

This module deliberately contains no Blender or GPU access. Host bakers use the
helpers to load the numerical boundary, build a Kernel, and prepare a completed
Cube for transfer to native resources.
"""
from functools import lru_cache
from importlib.resources import files
import numpy as np
from . import lut
from .bake import Kernel
from .bake_cache import SHARED, CachedKernel
from .boundary_polynomial import PolynomialTable
from .boundary_resource import load_data


@lru_cache(maxsize=1)
def resource():
    source = files(__package__) / 'd65_boundary.npz'
    with np.load(source, allow_pickle=False) as archive:
        data = load_data(archive)
    table = PolynomialTable(data)
    return data, table


def resources():
    data, table = resource()
    SHARED.reserve('CPU lookup', sum(v.nbytes for v in data.values()) +
                  sum(v.nbytes for v in table.expanded.values()))
    face_bytes = sum(v.nbytes for k, v in table.expanded.items() if k.startswith('face_'))
    SHARED.reserve('float64 contexts', 6 * (sum(v.nbytes for v in data.values()) + face_bytes))
    SHARED.reserve('lattices', (lut.SIZE**3 + lut.PREVIEW_SIZE**3) * 3 * 8 +
                   lut.SIZE**3 * 8 + lut.SIZE * 8)
    return data, table


def kernel(model):
    if len(model.grades) != 1 or model.directions != ('FORWARD',):
        raise ValueError('Regular forward Grade required')
    data, table = resources()
    forward = model.grades[0].forward
    return Kernel(data, forward.tone, forward.color, table, hk_guard=True)


def prepared(cube, encoded, checkpoint):
    checkpoint()
    cube.diagnostics.update(
        minimum_rgb=float(cube.values.min()),
        maximum_rgb=float(cube.values.max()),
        outside_cube_pixels=int(np.count_nonzero(
            np.any((cube.values < 0) | (cube.values > lut.MAXIMUM), axis=-1))),
    )
    cube.prepared_pixels = cube.packed_pixels() if encoded else cube.pixels()
    checkpoint()
    return cube
