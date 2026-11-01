"""Chunked analytic Grade/Stack evaluation of scene-linear float32 RGBA.

The same production kernels populate the preview/final LUTs. Raster baking
evaluates those kernels at every input pixel, without a lattice or interpolation.
The caller supplies the frozen scene-to-native and native-to-scene endpoints.
"""
import numpy as np

from . import baking, lut
from .export import input_repair

SCHEMA = 1
BLOCK = 16384


def grade_rgba(rgba, model, rows, *, stack=False, checkpoint=lambda: None,
               progress=lambda value: None, block=BLOCK):
    """Replace RGB in owned contiguous RGBA storage; leave alpha bits unchanged.

    Matrix products use float64 intermediates so finite float32 scene pixels
    can reach the established AP1 input projection without intermediate overflow.
    Per-pixel nonlinear computation uses the production float32 kernels.
    """
    if (type(rgba) is not np.ndarray or rgba.dtype != np.float32
            or rgba.ndim < 2 or rgba.shape[-1] != 4
            or not rgba.flags.c_contiguous or not rgba.flags.writeable):
        raise ValueError('Raster grading requires owned writable float32 RGBA storage')
    matrices = np.asarray(rows, dtype=np.float64)
    if matrices.shape != (2, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError('Raster grading requires two finite 3 by 3 endpoint matrices')
    if type(block) is not int or block < 1 or block > BLOCK:
        raise ValueError('Raster grading block exceeds the bounded pixel chunk')
    pixels = rgba.reshape(-1, 4)
    checkpoint()
    execution = model.prepare(checkpoint) if stack else baking.kernel(model)
    residual = 0.
    for start in range(0, len(pixels), block):
        checkpoint()
        chunk = pixels[start:start + block]
        if not np.isfinite(chunk).all():
            raise ValueError('Raster grading input contains nonfinite RGBA pixels')
        native = input_repair(chunk[:, :3].astype(np.float64) @ matrices[0].T,
                              lut.AP1_Y).astype(np.float32)
        if stack:
            graded = execution.bake_native(native, checkpoint=checkpoint)
            residual = max(residual, float(np.max(graded[:, 3], initial=0.)))
            graded = graded[:, :3]
        else:
            graded = execution.evaluate(native)
        checkpoint()
        with np.errstate(over='ignore', invalid='ignore'):
            result = (graded.astype(np.float64) @ matrices[1].T).astype(np.float32)
        if not np.isfinite(result).all():
            raise ArithmeticError('Raster grading output is not finite float32 RGB')
        chunk[:, :3] = result
        progress(min(start + len(chunk), len(pixels)) / max(1, len(pixels)))
    checkpoint()
    return dict(pixels=len(pixels), max_inverse_residual=residual)
