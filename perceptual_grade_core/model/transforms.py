"""Grade adapters and shared float32 forward/inverse Stack execution."""
from dataclasses import dataclass
import numpy as np
from . import linear, color, forward, lut
from threading import Lock


@dataclass(frozen=True)
class Settings:
    matrix: tuple
    tone: object
    color: object
    space: object

    @classmethod
    def from_parameters(cls, params, tone, color, space):
        return cls(linear.matrices(params, space)['xyz'], tone, color, space)


class Grade:
    def __init__(self, settings):
        self.settings=settings
        nonlinear=color.model(settings.tone,settings.color,settings.space)
        self.forward=forward.Model(nonlinear.tone,nonlinear,settings.matrix)

    def apply(self, rgb, checkpoint=None, export=True, entry=True):
        return self.forward.apply(rgb,checkpoint,export,entry)

    def apply_native(self, rgb, checkpoint=None, export=True, entry=True):
        return self.forward.apply_native(rgb,checkpoint,export,entry)


class Stack:
    prepared=False

    def __init__(self,entries,space='Linear Rec.709',directions=None):
        self.space=space;self.entries=tuple(entries)
        self.directions=tuple(directions) if directions is not None else ('FORWARD',)*len(self.entries)
        if len(self.directions)!=len(self.entries) or any(d not in ('FORWARD','INVERSE') for d in self.directions):
            raise ValueError('Stack directions must match entries and be FORWARD or INVERSE')
        self.grades=tuple(Grade(settings) for settings in self.entries)
        self.inverse_matrices=[];self._prepared=None;self._prepare_lock=Lock()
        for grade,direction in zip(self.grades,self.directions):
            if direction=='INVERSE':
                if np.linalg.matrix_rank(grade.forward.linear)<3:raise ValueError('Inverse Stage 1 matrix is singular')
                try:matrix=np.linalg.inv(grade.forward.matrix_native)
                except np.linalg.LinAlgError as e:raise ValueError('Inverse Stage 1 matrix is singular') from e
                if not np.isfinite(matrix).all():raise ValueError('Inverse Stage 1 matrix is not finite')
                self.inverse_matrices.append(matrix)
            else:self.inverse_matrices.append(None)
        self.input_matrix=self.grades[0].forward.matrix_native if self.directions and self.directions[0]=='FORWARD' else np.eye(3)
        self.output_matrix=self.inverse_matrices[-1] if self.directions and self.directions[-1]=='INVERSE' else np.eye(3)
        self.cache_key=('stack',13,space,tuple(zip(self.entries,self.directions)))
        from dataclasses import replace
        core=list(self.entries)
        if core and self.directions[0]=='FORWARD':core[0]=replace(core[0],matrix=linear.I3)
        if core and self.directions[-1]=='INVERSE':core[-1]=replace(core[-1],matrix=linear.I3)
        self.bake_key=('stack-core',13,space,tuple(zip(core,self.directions)))

    @staticmethod
    def _matrix(values,matrix):
        from ...color_core.scaled import Scalar as S,dot
        values=np.asarray(values)
        if np.array_equal(matrix,np.eye(3)):return values
        if values.dtype==object:
            return np.asarray([[S(row[1])*sum(axis)+dot(axis,[S(v)-row[1] for v in row]) for axis in matrix]
                               for row in values.reshape(-1,3)],dtype=object).reshape(values.shape)
        return values[...,1,None]*matrix.sum(axis=1)+(values-values[...,1,None])@matrix.T

    @property
    def is_prepared(self):return self._prepared is not None

    def prepare(self,checkpoint=None):
        """Prepare immutable NumPy kernels/tone tables once, before worker fanout."""
        if self._prepared is not None:return self._prepared
        from .baking import resources
        from .bake import Kernel
        from .stack_kernel import StackKernel
        with self._prepare_lock:
            if self._prepared is None:
                if checkpoint is not None:checkpoint()
                data,table=resources()
                kernels=[Kernel(data,g.forward.tone,g.forward.color,table,hk_guard=True) for g in self.grades]
                prepared=StackKernel(kernels,[g.forward.matrix_native for g in self.grades],
                    self.inverse_matrices,self.directions,lut.FROM_LUT,lut.AP1_Y,
                    [(g.forward.tone.settings,g.forward.color.params) for g in self.grades],checkpoint)
                if checkpoint is not None:checkpoint()
                self._prepared=prepared
        return self._prepared

    def bake_native(self,rgb,checkpoint=None):
        """Complete float32 core; native graph owns extracted endpoint matrices."""
        return self.prepare(checkpoint).bake_native(rgb,checkpoint=checkpoint)[...,:3]

    def apply_native(self,rgb,checkpoint=None):
        prepared=self.prepare(checkpoint)
        if checkpoint is not None:checkpoint()
        result=prepared.apply_native(rgb)
        if checkpoint is not None:checkpoint()
        return result

    def apply_cube(self,cube,rgb):
        before,after=lut.basis(self.space)
        native=self._matrix(np.asarray(rgb)@before.T,self.input_matrix)
        result=lut.tetrahedral(cube.values,native)
        return self._matrix(result,self.output_matrix)@after.T

    def apply(self,rgb,checkpoint=None):
        before,after=lut.basis(self.space)
        values=np.asarray(rgb,dtype='f4');before=np.asarray(before,dtype='f4');after=np.asarray(after,dtype='f4')
        native=values[...,1,None]+(values-values[...,1,None])@before.T
        result=self.apply_native(native,checkpoint)
        return result[...,1,None]+(result-result[...,1,None])@after.T
