"""Progressive Stack baking with shared CPU/GLSL math and native endpoints."""
import numpy as np
from . import baking, gpu, baker
from ...perceptual_grade_core.model import lut
from .bake_inverse_gpu import Program


class CPU(baking.CPU):
    def __init__(self,model,progressive=True,encoded=False):
        baker.Baker.__init__(self,model,progressive)
        self.encoded=encoded;self.calculated=False;self.execution=None

    def prepare(self):
        self.checkpoint()
        if self.execution is None:self.execution=self.model.prepare(self.checkpoint)
        return self.execution

    def evaluate_rows(self,rows):
        self.checkpoint()
        result=self.prepare().bake_native(lut.lattice()[rows],checkpoint=self.checkpoint)
        self.checkpoint()
        return result[:,:3]

    def advance(self):
        self.prepare()
        return super().advance()

    def finish(self):
        self.prepare()
        return super().finish()


class PreparedProgram:
    """Immutable per-grade uploads; all binds/dispatches stay on the main thread."""
    fields=('parameters','missing_tone_table','tone_lookup_bytes','parameter_bytes')

    def __init__(self,program,solvers):
        gpu.require_main_thread();self.program=program;self.bindings={}
        for solver in solvers:
            program.set_inverse(solver)
            self.bindings[id(solver)]=tuple(getattr(program,name) for name in self.fields)

    def set_inverse(self,solver):
        gpu.require_main_thread()
        for name,value in zip(self.fields,self.bindings[id(solver)]):setattr(self.program,name,value)

    def evaluate(self,values,operation):
        gpu.require_main_thread()
        return self.program.evaluate(values,operation)


class GPU(gpu.Baker,CPU):
    def __init__(self,model,program,progressive,encoded,device):
        gpu.require_main_thread();CPU.__init__(self,model,progressive,encoded)
        self.execution=self.prepare();self.program=PreparedProgram(program,self.execution.solvers)
        self.device=device;self.packing=[]

    def evaluate_rows(self,rows):
        gpu.require_main_thread();self.checkpoint()
        # Another prepared job can share the driver program, but each owns its
        # immutable bindings and rebinds before every stage of every batch.
        result=self.execution.bake_native(lut.lattice()[rows],backend='GPU',checkpoint=self.checkpoint,program=self.program)
        self.checkpoint()
        return result[:,:3]


def validate_program(program,execution):
    """Exercise forward and actual raw inverse computation before GPU adoption."""
    from ...perceptual_grade_core.model import color, tone, linear
    from ...perceptual_grade_core.model.bake import Kernel
    from ...perceptual_grade_core.model.bake_inverse import Inverse
    from ...perceptual_grade_core.model.tone_table import ToneTable
    k=execution.kernels[0]
    t=tone.model(tone.Parameters(input_midtones=45),linear.GradeSpace('D65'))
    c=color.Model(t,color.Parameters(global_gain=.15,web_hue_1_0=3.,web_chroma_1_2=.02,web_scale_2_4=.01,j_high_1=.1,adaptance_b_2=.01))
    kernel=Kernel(k.data,t,c,k.table,hk_guard=True)
    solver=Inverse(kernel,lut.FROM_LUT,lut.AP1_Y);solver.tone_table=ToneTable(t)
    points=np.array([[0,0,0],[.1,.2,.3],[.7,.2,.1],[.02,.08,.4],[2.6621823]*3],dtype='f4')
    program.set_inverse(solver)
    raw=kernel.grade(kernel.tone_state(kernel.enter(points)))
    for values,operation,expected in ((points,2,kernel.evaluate(points)),(raw,1,solver.evaluate(raw,True)[:,:3])):
        actual=program.evaluate(values,operation)[:,:3]
        limit=np.maximum(.001*np.max(abs(expected),axis=-1),1e-7)
        if (not np.isfinite(actual).all() or np.any(np.max(abs(actual-expected),axis=-1)>limit)
                or np.any(actual[np.max(abs(expected),axis=-1)==0]!=0)):
            raise ArithmeticError('GPU Stack conformance probe failed')
