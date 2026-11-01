"""Portable float32 Stack execution; adapters supply controls and matrices.

No Colour, Blender, UI or independent-oracle dependency is imported here.
CPU and GLSL carry raw appearance only across forward-to-inverse transitions.
"""
import time
import numpy as np
from .bake_inverse import Inverse
from .tone_table import ToneTable


class StackKernel:
    def __init__(self,kernels,matrices,inverses,directions,canonical,weights,keys,checkpoint=None,tone_factory=ToneTable):
        self.kernels=tuple(kernels);self.directions=tuple(directions)
        self.matrices=tuple(np.asarray(m,dtype='f4') for m in matrices)
        self.inverses=tuple(None if m is None else np.asarray(m,dtype='f4') for m in inverses)
        self.matching_keys=tuple(keys);self.weights=np.asarray(weights,dtype='f4')
        self.solvers=tuple(Inverse(k,canonical,weights) for k in self.kernels)
        tables={};self.tone_table_seconds=0.
        for solver,direction in zip(self.solvers,self.directions):
            if checkpoint is not None:checkpoint()
            tone=solver.kernel.tone
            if direction!='INVERSE' or not tone.active:continue
            key=(tuple(tone.knots),float(tone.width),float(tone.slope),tuple(tone.residual))
            if key not in tables:
                start=time.perf_counter();tables[key]=tone_factory(tone)
                self.tone_table_seconds+=time.perf_counter()-start
            solver.tone_table=tables[key]
        if checkpoint is not None:checkpoint()
        self.tone_tables=tuple(tables.values())
        self.input_matrix=self.matrices[0] if self.directions and self.directions[0]=='FORWARD' else np.eye(3,dtype='f4')
        self.output_matrix=self.inverses[-1] if self.directions and self.directions[-1]=='INVERSE' else np.eye(3,dtype='f4')
        self.program=None

    @staticmethod
    def matrix(rgb,matrix):
        return rgb[...,1,None]*matrix.sum(axis=1)+(rgb[...,0]-rgb[...,1])[...,None]*matrix[:,0]+(rgb[...,2]-rgb[...,1])[...,None]*matrix[:,2]

    def matching(self,left,right):
        return self.matching_keys[left]==self.matching_keys[right]

    def bake_native(self,values,backend='CPU',checkpoint=None,program=None):
        """RGB plus largest local inverse residual; endpoint matrices are outside."""
        if backend not in ('CPU','GPU'):raise ValueError('Unknown baking backend')
        result=np.asarray(values,dtype='f4')
        if result.shape[-1:]!=(3,) or not np.isfinite(result).all():raise ValueError('Stack input must contain finite RGB triples')
        shape=result.shape[:-1];result=result.reshape(-1,3).copy();errors=np.zeros(len(result),dtype='f4')
        pending=None;cancelled=False
        if backend == 'GPU' and self.solvers and program is None:
            raise ValueError('GPU execution requires a host-supplied compute program')
        if backend != 'GPU' or not self.solvers:
            program = None
        for i,(k,inverse,direction) in enumerate(zip(self.kernels,self.solvers,self.directions)):
            if checkpoint is not None:checkpoint()
            if direction=='FORWARD':
                if i:result=inverse.repair(self.matrix(result,self.matrices[i]))
                following=i+1<len(self.kernels) and self.directions[i+1]=='INVERSE'
                cancelled=following and self.matching(i,i+1)
                if cancelled:pending=result;continue
                if program is not None:
                    program.set_inverse(inverse);value=program.evaluate(result,3 if following else 2)
                    if following:pending=value
                    else:result=value[:,:3]
                elif following:pending=k.grade(k.tone_state(k.enter(result)))
                else:result=k.evaluate(result)
            else:
                if cancelled:result=pending
                else:
                    if program is not None:
                        program.set_inverse(inverse);value=program.evaluate(result if pending is None else pending,int(pending is not None))
                    else:value=inverse.evaluate(result if pending is None else pending,pending is not None)
                    result=value[:,:3];errors=np.maximum(errors,value[:,3])
                pending=None;cancelled=False
                if i+1<len(self.kernels):result=self.matrix(result,self.inverses[i])
        if checkpoint is not None:checkpoint()
        if not np.isfinite(result).all() or not np.isfinite(errors).all():raise ArithmeticError('Nonfinite Stack bake')
        return np.column_stack((result,errors)).reshape(*shape,4)

    def apply_native(self,values,backend='CPU'):
        result=np.asarray(values,dtype='f4')
        if not self.solvers:
            from .export import input_repair
            return input_repair(result,self.weights).astype('f4')
        result=self.solvers[0].repair(self.matrix(result,self.input_matrix))
        result=self.bake_native(result,backend)[...,:3]
        return self.matrix(result,self.output_matrix)
