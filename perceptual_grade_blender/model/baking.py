"""Blender-side regular Grade baker and native transfer adapter."""
import numpy as np
from ...perceptual_grade_core.model import baking as core_baking, lut
from . import baker
from ...perceptual_grade_core.model.bake_cache import SHARED, CachedKernel

resource = core_baking.resource
resources = core_baking.resources
kernel = core_baking.kernel
prepared = core_baking.prepared


class CPU(baker.Baker):
    """Incremental numerical job; its caller owns execution resources."""
    block = 16384
    device = 'NumPy float32'

    def __init__(self, model, progressive=True, encoded=False):
        super().__init__(model, progressive)
        self.kernel = kernel(model)
        self.cached = CachedKernel(self.kernel, lut.SIZE**3)
        self.encoded = encoded
        self.calculated = False

    def phase(self):
        shared, remaining = lut.refinement_indices()
        if self.progressive:
            if self.cursor < len(shared):
                return shared, self.cursor, 0
            return remaining, self.cursor - len(shared), len(shared)
        return None, self.cursor, 0

    def rows(self, phase, start, end):
        return np.arange(start, end) if phase is None else phase[start:end]

    def evaluate_rows(self, rows):
        self.checkpoint()
        values = self.cached.evaluate(lut.lattice()[rows], rows, self.checkpoint)
        self.checkpoint()
        if not np.isfinite(values).all():
            raise ArithmeticError('Float32 export is nonfinite')
        return values

    def complete(self):
        if self.progressive and self.cursor == lut.PREVIEW_SIZE**3:
            shared, _ = lut.refinement_indices()
            cube = lut.Cube(self.output[shared], lut.signature(self.model, lut.PREVIEW_SIZE),
                            self.space, lut.PREVIEW_SIZE)
            self.ready(cube, True)
        if self.cursor == lut.SIZE**3:
            self.ready(lut.Cube(self.output, self.signature, self.space), False)
            self.calculated = True

    def ready(self, cube, preview):
        cube.diagnostics = {'backend': self.backend, 'device': self.device}
        prepared(cube, self.encoded, self.checkpoint)
        if preview:
            self.preview = cube
        else:
            self.cube = cube

    @property
    def progress(self):
        return self.cursor / (lut.SIZE**3)

    def advance(self):
        self.checkpoint()
        if self.calculated:
            return False
        phase, begin, offset = self.phase()
        total = lut.SIZE**3 if phase is None else len(phase)
        end = min(begin + self.block, total)
        rows = self.rows(phase, begin, end)
        self.output[rows] = self.evaluate_rows(rows)
        self.cursor = offset + end
        self.complete()
        return not self.calculated

    def finish(self):
        while not self.calculated:
            self.advance()
        return self.cube


def clear():
    SHARED.clear()
    resource.cache_clear()
