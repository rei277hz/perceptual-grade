"""Incremental Blender-side LUT baker over portable numerical lattice helpers."""

from concurrent.futures import CancelledError
from threading import Event
import numpy as np

from ...perceptual_grade_core.model import lut

class Baker:
    """Incremental bake, optionally with a nested preview and disjoint refinement.

    No Blender objects. Each vertex evaluates the complete signed forward
    pipeline. Edits cancel worker jobs at numerical checkpoints.
    """
    backend='CPU'
    requires_main_thread=False

    def __init__(self,model,progressive=False):
        self.progressive=progressive
        self.restart(model)

    def restart(self,model):
        self._cancelled=Event()
        self._progress=0.
        self.space=lut.model_space(model)
        self.to_lut,self.from_lut=lut.basis(self.space)
        self.model=model;self.signature=lut.signature(model);self.cursor=0
        self.output=np.empty((lut.SIZE**3,3),np.float32)
        self.cube=None
        self.preview=None

    @property
    def progress(self):
        shared=lut.PREVIEW_SIZE**3 if self.preview is not None else 0
        return max(self._progress,(shared+self.cursor)/(lut.SIZE**3))

    def cancel(self):
        self._cancelled.set()

    def checkpoint(self,progress=None):
        if self._cancelled.is_set():raise CancelledError('Obsolete LUT bake')
        if progress is not None:self._progress=max(self._progress,min(.99,max(0,progress)))

    def advance(self):
        self.checkpoint()
        if self.cube is not None:return False
        if self.progressive and self.preview is None:
            shared,_=lut.refinement_indices()
            values=self.evaluate(lut.lattice()[shared],0,len(shared))
            self.output[shared]=values
            self.preview=lut.Cube(values,lut.signature(self.model,lut.PREVIEW_SIZE),self.space,lut.PREVIEW_SIZE)
            self.preview.diagnostics=getattr(self.model,'diagnostics',lambda:{})()
            return True
        block=lut.BLOCK
        remaining=lut.refinement_indices()[1] if self.progressive else None
        total=len(remaining) if remaining is not None else lut.SIZE**3
        begin=self.cursor;end=min(begin+block,total)
        rows=remaining[begin:end] if remaining is not None else slice(begin,end)
        offset=lut.PREVIEW_SIZE**3 if self.progressive else 0
        values=self.evaluate(lut.lattice()[rows],offset+begin,offset+end)
        self.output[rows]=values
        self.cursor=end
        if end==total:
            self.cube=lut.Cube(self.output,self.signature,self.space)
            self.cube.diagnostics=getattr(self.model,'diagnostics',lambda:{})()
        return self.cube is None

    def evaluate(self,native,start,stop):
        def checkpoint(progress=None):
            self.checkpoint(None if progress is None else (start+(stop-start)*progress)/(lut.SIZE**3))
        if getattr(self.model,'entries',None)==():
            # The empty composition is identity in the LUT's native basis.
            # Avoid losing exact cube edges in an unnecessary matrix round trip.
            values=np.asarray(native,dtype=np.float32)
        else:
            values=np.asarray(self.model.bake_native(native,checkpoint),dtype=np.float32)
        self.checkpoint()
        if not np.isfinite(values).all():raise ArithmeticError('LUT bake produced a nonfinite value')
        return values

    def finish_preview(self):
        if not self.progressive:raise ValueError('Preview requested for a final-only bake')
        while self.preview is None:self.advance()
        return self.preview

    def finish(self):
        while self.cube is None:self.advance()
        return self.cube
