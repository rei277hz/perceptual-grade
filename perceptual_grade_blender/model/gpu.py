"""Main-thread compute batches and numerical packing, without executor ownership."""
from threading import current_thread, main_thread
import numpy as np
from . import baking
from ...perceptual_grade_core.model import lut
from ...perceptual_grade_core.model.bake import Kernel
from .bake_gpu import Program, parameters
from ...perceptual_grade_core.model.bake_cache import SHARED
from .boundary_gpu import BATCH

def require_main_thread():
    if current_thread() is not main_thread():raise RuntimeError('GPU baking requires Blender main thread')


def validate_program(program,kernel):
    """Probe actual arithmetic, including a nonneutral grade and cube faces."""
    from ...perceptual_grade_core.model import color, tone, linear
    t=tone.model(tone.Parameters(input_midtones=45),linear.GradeSpace('D65'))
    c=color.Model(t,color.Parameters(global_gain=.5,web_hue_1_0=12,web_chroma_1_2=.2,web_scale_2_4=.1,j_a_1=5,adaptance_a_2=10))
    probe=Kernel(kernel.data,t,c,kernel.table,hk_guard=True)
    axis=np.array([0.,.002,.2,4.,128.,800.,1024.],dtype=np.float32)
    rgb=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    program.set_kernel(probe);actual=program.evaluate(rgb)[:,:3];expected=probe.evaluate(rgb)
    error=np.max(abs(actual.astype(float)-expected),-1)
    limit=np.maximum(.001*np.max(expected,-1),1e-7)
    if (not np.isfinite(actual).all() or np.any(error>limit)
            or np.any(actual[np.max(expected,-1)==0]!=0)):
        raise ArithmeticError('GPU float32 conformance probe failed')
    program.set_kernel(kernel)
    SHARED.reserve('GPU lookup',program.lookup_bytes)


class Baker(baking.CPU):
    backend='GPU'
    requires_main_thread=True
    block=BATCH

    def __init__(self,model,program,progressive,encoded,device):
        require_main_thread();super().__init__(model,progressive,encoded)
        self.program=program;self.device=device;self.packing=[]
        values=parameters(self.kernel)
        self.parameters=program.gpu.types.GPUTexture((256,1),format='R32F',
            data=program.gpu.types.Buffer('FLOAT',(1,256),values))

    def evaluate_rows(self,rows):
        require_main_thread();self.checkpoint();rgb=lut.lattice()[rows].astype(np.float32)
        if self.kernel.identity:return rgb
        self.program.parameters=self.parameters
        entry=self.cached.cached(('entry','GPU',self.device),rows,
            lambda missing:self.program.evaluate(rgb[missing],0))
        self.checkpoint();values=self.program.evaluate(entry,1)[:,:3]
        if not np.isfinite(values).all():raise ArithmeticError('GPU export is nonfinite')
        return values

    def ready(self,cube,preview):
        cube.diagnostics={'backend':self.backend,'device':self.device}
        baking.prepared(cube,self.encoded,self.checkpoint)
        if preview:self.preview=cube
        else:self.cube=cube

    def collect(self,wait=False):
        self.checkpoint()

    def advance(self):
        require_main_thread();self.collect();super().advance();self.collect()
        return not self.calculated

    def finish(self):
        require_main_thread()
        while not self.calculated:self.advance()
        self.collect(wait=True)
        return self.cube

    def cancel(self):
        super().cancel()


def clear():
    require_main_thread()
