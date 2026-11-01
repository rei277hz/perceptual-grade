"""Complete numerical LUT jobs, including GPU ownership and atlas preparation."""
import os
import time

from ...model import baking, gpu, stack_baking
from ....perceptual_grade_core.model import lut
from ....perceptual_grade_core.jobs.codec import thaw_model, freeze_cube, thaw_cube

_program=None
_stack_program=None
_initialized=False


def _device():
    global _initialized
    gpu.require_main_thread()
    import gpu as driver
    if not _initialized:
        driver.init();_initialized=True
    return driver.platform.backend_type_get()+' · '+driver.platform.renderer_get()


def _gpu_baker(model,progressive,encoded,stack,checkpoint):
    global _program,_stack_program
    device=_device()
    if stack:
        execution=model.prepare(checkpoint)
        if not execution.solvers:return stack_baking.CPU(model,progressive,encoded)
        if _stack_program is None:
            candidate=stack_baking.Program(execution.solvers[0])
            stack_baking.validate_program(candidate,execution)
            _stack_program=candidate
        return stack_baking.GPU(model,_stack_program,progressive,encoded,device)
    kernel=baking.kernel(model)
    if _program is None:
        candidate=gpu.Program(kernel)
        gpu.validate_program(candidate,kernel)
        _program=candidate
    return gpu.Baker(model,_program,progressive,encoded,device)


def _validate_math(kernel,parameters):
    """Exercise the actual shared shader, including compensated power math."""
    import numpy as np
    from ....perceptual_grade_core.model import bake as numerical
    from ...model import bake_gpu, boundary_gpu
    def source(data,offsets,expanded):
        return bake_gpu.source(data,offsets,expanded).split('void main()',1)[0]+'''
void main() {
    int id=int(gl_GlobalInvocationID.x);if(id>=sample_count||operation<0)return;
    ivec2 address=ivec2(id%256,id/256);vec4 v=texelFetch(input_pixels,address,0);
    imageStore(output_pixels,address,vec4(pg_power(v.x,v.y),pg_atan2(v.z,v.w),0.0,0.0));
}
'''
    program=boundary_gpu.Program(kernel.data,source,True,kernel.table.expanded)
    program.parameters=parameters
    bases=np.r_[np.float32(0),np.geomspace(1e-12,1e4,513).astype('f4'),np.float32(1)]
    exponents=np.array([.42,numerical.GAMMA,1/numerical.GAMMA,50/21],dtype='f4')
    base,exponent=np.meshgrid(bases,exponents);base=base.ravel();exponent=exponent.ravel()
    angles=np.linspace(-np.pi,np.pi,len(base));y=np.sin(angles).astype('f4');x=np.cos(angles).astype('f4')
    y[:8]=[0,-0.,1,-1,1e-20,-1e-20,1,-1];x[:8]=[-1,-1,0,0,1,1,-1e-20,-1e-20]
    actual=program.evaluate(np.column_stack((base,exponent,y,x)))
    expected=np.power(base.astype(float),exponent.astype(float)).astype('f4')
    if not np.isfinite(actual).all():raise ArithmeticError('GPU arithmetic probe is nonfinite')
    ulp=np.abs(actual[:,0].astype('f4').view('u4').astype('i8')-expected.view('u4').astype('i8'))
    if np.max(ulp)>3 or np.any(actual[base==0,0]!=0):
        raise ArithmeticError('GPU power conformance exceeds 3 ULP')
    expected_angle=np.arctan2(y.astype(float),x.astype(float)).astype('f4')
    if np.any(abs(actual[:,1]-expected_angle)>2.4e-7+2e-7*abs(expected_angle)):
        raise ArithmeticError('GPU angle conformance failed')
    return dict(power_max_ulp=int(np.max(ulp)),samples=len(base))


def handle_probe(inputs,context):
    """A GPU-only capability receipt; failure must never become CPU success."""
    global _program,_stack_program
    if context.role!='gpu':raise RuntimeError('GPU probe requires the GPU helper')
    from ....perceptual_grade_core.model import color, linear, tone, transforms
    started=time.perf_counter();context.checkpoint()
    settings=transforms.Settings(linear.I3,tone.Parameters(input_midtones=45),
        color.Parameters(global_gain=.2,web_hue_1_4=12.,web_chroma_1_4=.05),linear.GradeSpace('D65'))
    model=transforms.Stack((settings,))
    try:
        regular=_gpu_baker(model,False,False,False,context.checkpoint)
        context.checkpoint()
        _gpu_baker(model,False,False,True,context.checkpoint)
        context.checkpoint()
        numerical=_validate_math(regular.kernel,_program.parameters)
        context.checkpoint()
    except BaseException:
        _program=_stack_program=None
        raise
    return dict(conformance=True,backend='GPU',device=regular.device,worker_pid=os.getpid(),
        checks=['forward','inverse','power','angle'],numerical=numerical,
        probe_seconds=time.perf_counter()-started)


def handle(inputs,context):
    started=time.perf_counter()
    context.checkpoint()
    model=thaw_model(inputs['model'])
    progressive=inputs.get('progressive',True);encoded=inputs.get('encoded',False)
    stack=inputs.get('stack',False)
    program_reused=context.role=='gpu' and (_stack_program if stack else _program) is not None
    if context.role=='gpu':
        baker=_gpu_baker(model,progressive,encoded,stack,context.checkpoint)
    else:
        baker=(stack_baking.CPU if stack else baking.CPU)(model,progressive,encoded)
    checkpoint=baker.checkpoint
    def check(progress=None):
        context.checkpoint();checkpoint(progress)
    baker.checkpoint=check
    prepared=time.perf_counter()
    ready=baker.ready
    def pack_ready(cube,preview):
        calculated=time.perf_counter()
        ready(cube,preview)
        packed=time.perf_counter()
        program=getattr(baker,'program',None)
        program=getattr(program,'program',program)
        cube.diagnostics.update(worker_pid=os.getpid(),worker_started_at=started,
            worker_preparation_s=prepared-started,worker_calculated_at=calculated,
            worker_packed_at=packed,worker_packing_s=packed-calculated,
            worker_cache_bytes=baking.SHARED.bytes,
            worker_program_reused=program_reused,
            shader_creation_s=float(getattr(program,'compile_seconds',0.)))
    baker.ready=pack_ready
    preview=None
    while baker.cube is None:
        context.checkpoint();baker.advance()
        if baker.preview is not None and baker.preview is not preview:
            preview=baker.preview
            preview.diagnostics['worker_pid']=os.getpid()
            context.preview(freeze_cube(preview))
        context.progress(baker.progress)
    baker.cube.diagnostics['worker_pid']=os.getpid()
    context.checkpoint()
    return freeze_cube(baker.cube)


def pack(inputs,context):
    context.checkpoint();cube=thaw_cube(inputs['cube'])
    if inputs.get('preview'):
        preview=lut.Cube(cube.values[::2,::2,::2],inputs['preview_signature'],cube.space,lut.PREVIEW_SIZE)
        preview.diagnostics=dict(cube.diagnostics)
        preview.prepared_pixels=preview.packed_pixels() if inputs['encoded'] else preview.pixels()
        context.checkpoint();return freeze_cube(preview)
    pixels=cube.packed_pixels() if inputs['encoded'] else cube.pixels()
    context.checkpoint();return pixels
