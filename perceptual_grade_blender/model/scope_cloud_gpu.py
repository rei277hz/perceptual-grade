"""Stage 3 Warp cloud rasterisation.

The public result of this module is a small raster, rather than the sampled
appearance arrays used by the numerical scope evaluator.  The CPU renderer is
also the correctness fallback for helpers without a usable graphics context.
GPU ownership is deliberately kept local to the helper process; callers only
receive ``uint8`` coverage tiles.
"""

from __future__ import annotations

import time
from typing import Any, Iterable
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


CACHE_LIMIT = 4
_program = None
_raster_shader = None
_raster_batch = None
_cache = OrderedDict()


class UnsupportedGPUCloud(ArithmeticError):
    """This input needs CPU arithmetic; the graphics device is still healthy."""

from ...perceptual_grade_core.model.scope_cloud import POINT_SIZE, MAX_OPACITY, TONES, CPU_RASTER_CANDIDATES, _descriptor, _field, _tone_value, _primary, _comparison, _raster_layer, _cpu_render



# Both computation and rasterization consume textures. Neither path expands a
# point into a CPU mesh or reads an intermediate coordinate texture back.
COMPUTE = r'''
vec4 pg_cloud_project(vec4 state) {
    if(any(isnan(state)) || any(isinf(state)))return vec4(0.0);
    vec2 ab=state.yz;
    float root=sqrt(abs(state.w));
    float scale=max(max(abs(ab.x),abs(ab.y)),root);
    vec2 result=vec2(0.0);
    if(scale>0.0) {
        vec3 normalized=vec3(ab,root)/scale;
        result=normalized.xy/length(normalized);
    }
    return vec4(result,state.w<0.0?1.0:0.0,1.0);
}
void main() {
    int id=int(gl_GlobalInvocationID.x);if(id>=sample_count)return;
    ivec2 address=ivec2(id%256,id/256);
    vec4 state=texelFetch(input_pixels,address,0);
    if(operation==1) {
        vec3 rgb;
        state=pg_export_state(pg_hk_guard(pg_grade(state),state.w),rgb);
    }
    imageStore(output_pixels,address,pg_cloud_project(state));
}
'''


def source(data, offsets, expanded):
    """Shared grade equations, with a post-tone Stage 3-only entry point."""
    from . import bake_gpu
    return bake_gpu.source(data, offsets, expanded).split('void main()', 1)[0] + COMPUTE


@dataclass
class _Layer:
    points: object
    weights: object
    mass: object
    count: int
    target: int


@dataclass
class _Cached:
    # The strong reference prevents id reuse while GPU resources are cached.
    result: object
    layers: tuple


def _texture(values):
    import gpu
    values=np.asarray(values,dtype=np.float32).reshape(-1,4)
    height=max(1,(len(values)+255)//256)
    if height>gpu.capabilities.max_texture_size_get():
        raise UnsupportedGPUCloud('Warp cloud sample texture exceeds GPU limits')
    padded=np.zeros((height,256,4),dtype=np.float32)
    padded.reshape(-1,4)[:len(values)]=values
    return gpu.types.GPUTexture((256,height),format='RGBA32F',
        data=gpu.types.Buffer('FLOAT',padded.shape,padded))


def _state(q,energy):
    values=np.concatenate((np.asarray(q).reshape(-1,3),np.asarray(energy).reshape(-1,1)),axis=1)
    if values.dtype==object:
        raise UnsupportedGPUCloud('Extended-range cloud appearance requires CPU raster')
    # Float32 GPU arithmetic must not silently erase a nonzero signed branch.
    # Fall back for extreme appearances, before any GPU coordinates are made.
    finite=np.isfinite(values)
    if np.any(finite & ((abs(values)>2.**60) | ((abs(values)>0) & (abs(values)<2.**-60)))):
        raise UnsupportedGPUCloud('Cloud appearance exceeds the GPU arithmetic range')
    return values.astype(np.float32)


def _parameters(model):
    from . import baking, bake_gpu
    from ...perceptual_grade_core.model.bake import Kernel
    p=model.settings
    if (np.max(abs(np.asarray(p.lightness)))>1.e6 or
        np.max(abs(np.asarray(p.adaptance)))>32 or
        np.max(abs(np.asarray(p.chroma)))>16 or abs(p.global_gain)>16 or
        np.max(abs(np.asarray(p.web.log_chroma)))>16 or
        np.max(abs(np.asarray(p.web.log_scale)))>16):
        raise UnsupportedGPUCloud('Cloud controls require extended-range CPU evaluation')
    data,table=baking.resource()
    kernel=Kernel(data,model.tone,model,table,hk_guard=True)
    parameters=bake_gpu.parameters(kernel)
    if not np.isfinite(parameters).all():
        raise UnsupportedGPUCloud('Cloud parameters exceed finite GPU range')
    return kernel


def _prepare(result,checkpoint=None):
    global _program
    import gpu
    from . import boundary_gpu, bake_gpu
    from .gpu import require_main_thread
    require_main_thread()
    started=time.perf_counter()
    key=id(result)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key].layers,dict(preparation_seconds=0.,compute_submission_seconds=0.,point_cache_hit=1)
    selected=result['_colors'];model=selected[0]['model']
    kernel=_parameters(model)
    if _program is None:
        _program=boundary_gpu.Program(kernel.data,source,True,kernel.table.expanded)
    bake_gpu.Program.set_kernel(_program,kernel)
    direct=('stage2_appearance' in result and 'stack_output_rgb' not in result and
            'grade_output_rgb' not in result and (model.active or model.tone.active))
    if direct:
        appearance=result['stage2_appearance']
        primary=_state(appearance.q,appearance.ja*abs(appearance.ja))
    else:
        primary=_state(selected[0]['after'],selected[0]['after_energy'])
    values=[(primary,np.stack([s.get('opacity_weights',s['weights']) for s in selected],axis=-1),
             selected[0].get('sample_mass'),int(direct))]
    comparisons=result.get('_comparisons_selected')
    if comparisons:
        signal=comparisons[0]['signal']
        values.append((_state(signal['appearance'],signal['energy']),
                       np.stack([s.get('opacity_weights',s['weights']) for s in comparisons],axis=-1),
                       comparisons[0].get('sample_mass'),0))
    layers=[];submitted=0.;chunk_size=65536
    for target,(state,weights,mass,operation) in enumerate(values):
        if checkpoint is not None:checkpoint()
        weights=np.asarray(weights,dtype=np.float32).reshape(-1,4)
        if len(weights)!=len(state):raise ValueError('Cloud appearance and membership sizes differ')
        weights=np.where(np.isfinite(weights),np.clip(weights,0.,1.),0.)
        population=np.zeros((len(state),4),np.float32)
        population[:,0]=1. if mass is None else np.asarray(mass).reshape(-1)
        if not np.isfinite(population).all():raise UnsupportedGPUCloud('Cloud sample population exceeds float32 range')
        max_height=gpu.capabilities.max_texture_size_get()
        batch=min(chunk_size,max_height*256)
        for start in range(0,max(1,len(state)),batch):
            part=state[start:start+batch];members=weights[start:start+batch]
            pixels=_texture(part);memberships=_texture(members);masses=_texture(population[start:start+batch])
            output=gpu.types.GPUTexture((256,max(1,(len(part)+255)//256)),format='RGBA32F')
            shader=_program.shader;shader.bind()
            shader.uniform_sampler('input_pixels',pixels)
            shader.uniform_sampler('boundary_cache',_program.cache)
            shader.uniform_sampler('face_atlas',_program.face_atlas)
            shader.uniform_sampler('grade_parameters',_program.parameters)
            shader.uniform_int('operation',operation)
            shader.uniform_int('sample_count',len(part));shader.image('output_pixels',output)
            begin=time.perf_counter()
            if len(part):gpu.compute.dispatch(shader,(len(part)+63)//64,1,1)
            submitted+=time.perf_counter()-begin
            layers.append(_Layer(output,memberships,masses,len(part),target))
    entry=_Cached(result,tuple(layers));_cache[key]=entry
    while len(_cache)>CACHE_LIMIT:_cache.popitem(last=False)
    return entry.layers,dict(preparation_seconds=time.perf_counter()-started-submitted,
                            compute_submission_seconds=submitted,point_cache_hit=0)


def _raster():
    global _raster_shader,_raster_batch
    import gpu
    from gpu_extras.batch import batch_for_shader
    if _raster_shader is None:
        interface=gpu.types.GPUStageInterfaceInfo('pg_scope_cloud_raster')
        interface.smooth('VEC2','pointUV')
        interface.smooth('VEC2','canvasPosition')
        interface.flat('FLOAT','pointOpacity')
        interface.flat('FLOAT','pointNegative')
        interface.flat('FLOAT','pointMass')
        info=gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0,'VEC2','corner');info.vertex_out(interface)
        info.sampler(0,'FLOAT_2D','cloud_points')
        info.sampler(1,'FLOAT_2D','cloud_weights')
        info.sampler(2,'FLOAT_2D','cloud_mass')
        info.push_constant('VEC4','plotRect');info.push_constant('VEC4','limits')
        info.push_constant('VEC4','target') # atlas w,h and tile origin x,y
        info.push_constant('FLOAT','pointRadius');info.push_constant('INT','tone')
        info.fragment_out(0,'VEC4','fragColor')
        info.vertex_source(r'''
void main() {
    ivec2 address=ivec2(gl_InstanceID%256,gl_InstanceID/256);
    vec4 point=texelFetch(cloud_points,address,0);
    float weight=texelFetch(cloud_weights,address,0)[tone];
    vec2 chart=vec2(-point.y,point.x);
    vec2 screen=plotRect.xy+(chart-limits.xy)*plotRect.zw/(limits.zw-limits.xy);
    screen+=corner*pointRadius*(1.0+point.z);
    gl_Position=vec4(2.0*(screen+target.zw)/target.xy-1.0,0.0,1.0);
    pointUV=corner;canvasPosition=screen;
    pointOpacity=weight*point.w;pointNegative=point.z;
    pointMass=texelFetch(cloud_mass,address,0).x;
}
''')
        info.fragment_source(r'''
void main() {
    if(canvasPosition.x<plotRect.x || canvasPosition.y<plotRect.y ||
       canvasPosition.x>=plotRect.x+plotRect.z || canvasPosition.y>=plotRect.y+plotRect.w)
        discard;
    float radius2=dot(pointUV,pointUV);
    if(radius2>1.0 || (pointNegative>0.5 && radius2<0.25))discard;
    float alpha=0.35*clamp(pointOpacity,0.0,1.0);
    float coverage=pointMass==1.0?alpha:1.0-pow(1.0-alpha,max(pointMass,0.0));
    fragColor=vec4(1.0,0.0,0.0,coverage);
}
''')
        _raster_shader=gpu.shader.create_from_info(info)
        corners=np.array([(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)],dtype=np.float32)
        _raster_batch=batch_for_shader(_raster_shader,'TRIS',dict(corner=corners))
    return _raster_shader,_raster_batch


def _gpu_render(result,views,checkpoint=None):
    import gpu
    layers,preparation=_prepare(result,checkpoint)
    shader,batch=_raster();records=[]
    for view in views:
        if checkpoint is not None:checkpoint()
        width,height=view['width'],view['height']
        layer_count=max((layer.target for layer in layers),default=-1)+1
        atlas_width,atlas_height=4*width,max(1,layer_count)*height
        if max(atlas_width,atlas_height)>gpu.capabilities.max_texture_size_get():
            raise UnsupportedGPUCloud('Native Warp cloud raster exceeds GPU texture limits')
        coverage=gpu.types.GPUTexture((atlas_width,atlas_height),format='R32F')
        framebuffer=gpu.types.GPUFrameBuffer(color_slots=(coverage,))
        blend=gpu.state.blend_get();viewport=gpu.state.viewport_get()
        depth=gpu.state.depth_test_get();scissor=gpu.state.scissor_get()
        begin=time.perf_counter()
        try:
            with framebuffer.bind():
                gpu.state.viewport_set(0,0,atlas_width,atlas_height)
                gpu.state.depth_test_set('NONE')
                gpu.state.scissor_set(0,0,atlas_width,atlas_height)
                framebuffer.clear(color=(0.,0.,0.,0.))
                gpu.state.blend_set('ALPHA')
                shader.bind();shader.uniform_float('plotRect',view['rect'])
                shader.uniform_float('pointRadius',POINT_SIZE*view['scale']/2.)
                for layer in layers:
                    shader.uniform_sampler('cloud_points',layer.points)
                    shader.uniform_sampler('cloud_weights',layer.weights)
                    shader.uniform_sampler('cloud_mass',layer.mass)
                    for tone in range(4):
                        shader.uniform_int('tone',tone)
                        shader.uniform_float('limits',view['bounds'][tone])
                        shader.uniform_float('target',(atlas_width,atlas_height,tone*width,layer.target*height))
                        if layer.count:batch.draw_instanced(shader,instance_count=layer.count)
                raster=time.perf_counter()-begin
                begin=time.perf_counter()
                # This is the only production GPU readback. Accumulation is
                # R32F; conversion to UBYTE happens once, at the handoff.
                values=framebuffer.read_color(0,0,atlas_width,atlas_height,1,0,'UBYTE')
                pixels=np.array(values,dtype=np.uint8).reshape(max(1,layer_count),height,4,width)[:layer_count]
                pixels=np.ascontiguousarray(pixels.transpose(0,2,1,3))
                readback=time.perf_counter()-begin
        finally:
            gpu.state.blend_set(blend);gpu.state.viewport_set(*viewport)
            gpu.state.depth_test_set(depth);gpu.state.scissor_set(*scissor)
        timings=dict(preparation,raster_submission_seconds=raster,readback_seconds=readback)
        records.append(dict(view=view,pixels=pixels,backend='gpu',timings=timings))
        preparation=dict(preparation_seconds=0.,compute_submission_seconds=0.,point_cache_hit=1)
    return tuple(records)


def render(result: Any, views: Iterable[Any] = (), use_gpu: bool = True, checkpoint=None) -> tuple[dict[str, Any], ...]:
    """Return four-tone rasters; device failures use the runtime bulk retry.

    All rich numerical state stays in the helper. The runtime retries a GPU
    failure on its CPU lane by calling this function with ``use_gpu=False``.
    Expected input/raster range limits use local CPU math without marking the
    shared GPU lane unhealthy or disabling unrelated LUT jobs.
    """
    normalized=tuple(_descriptor(view) for view in views)
    if not normalized:return ()
    if checkpoint is not None:checkpoint()
    if not use_gpu:return _cpu_render(result,normalized,checkpoint)
    try:
        return _gpu_render(result,normalized,checkpoint)
    except UnsupportedGPUCloud as error:
        records=_cpu_render(result,normalized,checkpoint)
        for record in records:record['fallback_reason']=str(error)
        return records


def clear():
    """Release resources from the owning helper's graphics thread."""
    global _program,_raster_shader,_raster_batch
    _cache.clear();_program=None;_raster_shader=None;_raster_batch=None
