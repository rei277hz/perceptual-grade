"""Helper GPU evaluation, raster reduction and bounded statistical gathering.

No full point/state texture is read back. Numerical readbacks are one 240-float
input diagnostic (two 96-bin histograms and 48 hue sectors), plus at most 512
pairs for the helper's robust fit.
"""
import time
import numpy as np
from . import scope_cloud_gpu as cloud

_engine=None
_density_shader=None
_density_batch=None
_row_shader=None
_max_shader=None
_quant_shader=None
_gather_shader=None

COMPUTE=r'''
void main() {
    int id=int(gl_GlobalInvocationID.x);if(id>=sample_count)return;
    ivec2 at=ivec2(id%256,id/256);
    vec4 state=texelFetch(input_pixels,at,0);
    vec3 rgb=texelFetch(input_rgb,at,0).xyz;
    if(operation==1) {
        state=pg_export_state(pg_hk_guard(pg_grade(state),state.w),rgb);
        rgb=pg_native(rgb,native_first,native_last);
    }
    if(operation==2) {
        float c=length(state.yz),r=PG_CHROMA_CROSSOVER*pg_physical_capacity(state.w,state.yz);
        float scale=max(c,r),cn=pg_div(c,scale),rn=pg_div(r,scale),den=cn*cn+rn*rn;
        float visibility=c/length(vec2(c,2.0));visibility*=visibility;
        float hue=c>0.0?atan(state.z,state.y):0.0;
        state=vec4(visibility*pg_div(rn*rn,den),visibility*pg_div(cn*cn,den),sqrt(abs(state.w)+66.0*c),hue);
    }
    imageStore(output_state,at,state);
    imageStore(output_rgb,at,vec4(rgb,1.0));
    imageStore(output_cloud,at,pg_cloud_project(state));
}
'''


class Engine:
    def __init__(self,model):
        import gpu
        from . import boundary_gpu, bake_gpu
        self.gpu=gpu
        kernel=cloud._parameters(model)
        values,offsets=boundary_gpu.pack(kernel.data,kernel.table.expanded)
        codes=kernel.table.expanded['face_codes'];width=codes.shape[-1]
        atlas=codes.reshape(-1,width).astype(np.float32)/65535.
        self.cache=gpu.types.GPUTexture((256,len(values)),format='RGBA32F',
            data=gpu.types.Buffer('FLOAT',values.shape,values))
        self.faces=gpu.types.GPUTexture((width,len(atlas)),format='R16',
            data=gpu.types.Buffer('FLOAT',atlas.shape,atlas))
        info=gpu.types.GPUShaderCreateInfo();info.local_group_size(64,1,1)
        for slot,name in enumerate(('input_pixels','boundary_cache','grade_parameters','face_atlas','input_rgb')):
            info.sampler(slot,'FLOAT_2D',name)
        for slot,name in enumerate(('output_state','output_rgb','output_cloud')):
            info.image(slot,'RGBA32F','FLOAT_2D',name,qualifiers={'WRITE'})
        info.push_constant('INT','operation');info.push_constant('INT','sample_count')
        info.push_constant('VEC3','native_first');info.push_constant('VEC3','native_last')
        source=bake_gpu.source(kernel.data,offsets,kernel.table.expanded).split('void main()',1)[0]
        info.compute_source(source+cloud.COMPUTE.split('void main()',1)[0]+COMPUTE)
        self.shader=gpu.shader.create_from_info(info)
        self.model=None
        self.configure(model,kernel)

    def configure(self,model,kernel=None):
        from . import bake_gpu
        if self.model is model:return
        params=bake_gpu.parameters(cloud._parameters(model) if kernel is None else kernel)
        self.parameters=self.gpu.types.GPUTexture((256,1),format='R32F',
            data=self.gpu.types.Buffer('FLOAT',(1,256),params))
        self.model=model

    def evaluate(self,model,inputs,operation):
        from ...perceptual_grade_core.model import linear
        from ...perceptual_grade_core.model.forward import Model
        gpu=self.gpu;shader=self.shader;count=inputs['count']
        self.configure(model)
        height=max(1,(count+255)//256)
        state=gpu.types.GPUTexture((256,height),format='RGBA32F')
        rgb=gpu.types.GPUTexture((256,height),format='RGBA32F')
        points=gpu.types.GPUTexture((256,height),format='RGBA32F')
        engine=Model(model.tone,model)
        rec709=(np.eye(3) if isinstance(model.tone.space,linear.GradeSpace) else
                np.asarray(linear.xyz_to_rgb('Linear Rec.709'))@np.asarray(linear.rgb_to_xyz(model.tone.space)))
        native=rec709@np.linalg.inv(engine.to_acescg)
        shader.bind()
        for name,value in (('input_pixels',inputs['state']),('input_rgb',inputs['rgb']),
            ('boundary_cache',self.cache),('face_atlas',self.faces),('grade_parameters',self.parameters)):
            shader.uniform_sampler(name,value)
        shader.uniform_int('sample_count',count);shader.uniform_int('operation',operation)
        shader.uniform_float('native_first',native[:,0]);shader.uniform_float('native_last',native[:,2])
        for name,value in (('output_state',state),('output_rgb',rgb),('output_cloud',points)):shader.image(name,value)
        if count:gpu.compute.dispatch(shader,(count+63)//64,1,1)
        return dict(state=state,rgb=rgb,points=points,meta=inputs['meta'],mass=inputs['mass'],count=count,width=inputs['width'])


def _inputs(signal,space):
    from .scope_active import _rec709
    state=cloud._texture(cloud._state(signal['q'],signal['energy']))
    rgb=np.zeros((len(signal['q']),4),np.float32);rgb[:,:3]=_rec709(signal['rgb'],space)
    if not np.isfinite(rgb).all():raise cloud.UnsupportedGPUCloud('Scope RGB exceeds float32 range')
    meta=np.zeros_like(rgb);meta[:,0]=signal['alpha'];meta[:,1]=signal['x'];meta[:,2]=signal.get('mass',1.)
    population=np.zeros_like(rgb);population[:,0]=meta[:,2]
    if not np.isfinite(meta).all():raise cloud.UnsupportedGPUCloud('Scope sample metadata exceeds float32 range')
    return dict(state=state,rgb=cloud._texture(rgb),meta=cloud._texture(meta),mass=cloud._texture(population),count=len(rgb),width=signal['width'])


def evaluate(base,model,selected,checkpoint=None):
    global _engine
    from .gpu import require_main_thread
    require_main_thread()
    if checkpoint is not None:checkpoint()
    if _engine is None:_engine=Engine(model)
    else:_engine.configure(model)
    if selected['gpu'] is None:
        value=_inputs(selected,model.tone.space)
        value['comparison']=None if selected['comparison'] is None else _inputs(selected['comparison'],model.tone.space)
        value['drivers']=None
        value['reference_output']=None
        # The sample identities are stable for the lifetime of this selected
        # group.  Keep the index texture and gather destination here so each
        # Stage 3 update only dispatches and reads the new state texture.
        value['gather_cache']={}
        selected['gpu']=value
    inputs=selected['gpu']
    operation=int('stack_output_rgb' not in base and (model.tone.active or model.active))
    result=_engine.evaluate(model,inputs,operation)
    result['comparison']=None
    if inputs['comparison'] is not None:
        # Reference is an independent, unchanged signal at the primary UVs.
        if inputs['reference_output'] is None:inputs['reference_output']=_engine.evaluate(model,inputs['comparison'],0)
        result['comparison']=inputs['reference_output']
    result['backend']='gpu'
    result['_gather_cache']=inputs['gather_cache']
    return result


def warp(output,view,tone,checkpoint=None):
    import gpu
    if checkpoint is not None:checkpoint()
    shader,batch=cloud._raster();layers=[output]
    if output['comparison'] is not None:layers.append(output['comparison'])
    width,height=view['width'],view['height'];atlas_height=len(layers)*height
    if max(width,atlas_height)>gpu.capabilities.max_texture_size_get():
        raise cloud.UnsupportedGPUCloud('Native scope target exceeds GPU limits')
    target=gpu.types.GPUTexture((width,atlas_height),format='R32F')
    framebuffer=gpu.types.GPUFrameBuffer(color_slots=(target,))
    saved=_state();start=time.perf_counter()
    try:
        with framebuffer.bind():
            _viewport(width,atlas_height);framebuffer.clear(color=(0.,0.,0.,0.));gpu.state.blend_set('ALPHA')
            shader.bind();shader.uniform_float('plotRect',view['rect']);shader.uniform_float('limits',view['bounds'][tone])
            shader.uniform_float('pointRadius',cloud.POINT_SIZE*view['scale']/2.);shader.uniform_int('tone',0)
            for i,layer in enumerate(layers):
                shader.uniform_sampler('cloud_points',layer['points']);shader.uniform_sampler('cloud_weights',layer['meta'])
                shader.uniform_sampler('cloud_mass',layer['mass'])
                shader.uniform_float('target',(width,atlas_height,0,i*height))
                if layer['count']:batch.draw_instanced(shader,instance_count=layer['count'])
            raster=time.perf_counter()-start;start=time.perf_counter()
            pixels=np.array(framebuffer.read_color(0,0,width,atlas_height,1,0,'UBYTE'),np.uint8)
            pixels=pixels.reshape(len(layers),1,height,width)
            readback=time.perf_counter()-start
    finally:_restore(saved)
    return dict(view=view,tone=tone,pixels=pixels,backend='gpu',
        timings=dict(raster_submission_seconds=raster,readback_seconds=readback))


def _state():
    import gpu
    return (gpu.state.blend_get(),gpu.state.viewport_get(),gpu.state.depth_test_get(),gpu.state.scissor_get())


def _restore(state):
    import gpu
    blend,viewport,depth,scissor=state
    gpu.state.blend_set(blend);gpu.state.viewport_set(*viewport)
    gpu.state.depth_test_set(depth);gpu.state.scissor_set(*scissor)


def _viewport(width,height):
    import gpu
    gpu.state.viewport_set(0,0,width,height);gpu.state.scissor_set(0,0,width,height);gpu.state.depth_test_set('NONE')


def _density():
    global _density_shader,_density_batch
    if _density_shader is not None:return _density_shader,_density_batch
    import gpu
    from gpu_extras.batch import batch_for_shader
    interface=gpu.types.GPUStageInterfaceInfo('pg_active_density')
    interface.flat('FLOAT','amount')
    info=gpu.types.GPUShaderCreateInfo();info.vertex_in(0,'VEC2','corner');info.vertex_out(interface)
    for slot,name in enumerate(('states','rgb_values','meta_values')):info.sampler(slot,'FLOAT_2D',name)
    info.push_constant('INT','kind');info.push_constant('INT','source_width')
    info.push_constant('FLOAT','maximum');info.push_constant('VEC2','target_size')
    info.fragment_out(0,'VEC4','fragColor')
    info.vertex_source(r'''
void main() {
    ivec2 at=ivec2(gl_InstanceID%256,gl_InstanceID/256);
    vec4 q=texelFetch(states,at,0),meta=texelFetch(meta_values,at,0);
    vec2 cell=vec2(-1000.0);amount=meta.x*meta.z;
    if(kind==0) {
        vec3 rgb=texelFetch(rgb_values,at,0).xyz,a=abs(rgb);
        vec3 code=sign(rgb)*mix(1.099*pow(a,vec3(.45))-.099,4.5*a,lessThan(a,vec3(.018)));
        float y=dot(code,vec3(.2126,.7152,.0722));vec2 uv=vec2((code.b-y)/1.8556,(code.r-y)/1.5748);
        if(!any(isnan(uv))&&!any(isinf(uv))&&all(lessThanEqual(abs(uv),vec2(.65))))
            cell=min(vec2(143),floor((uv+.65)*(144.0/1.3)));
    } else if(kind==1) {
        if(!isnan(q.x)&&!isinf(q.x)&&q.x>=0.0&&q.x<=maximum)
            cell=vec2(144+min(191.0,floor(meta.y*192.0/float(source_width))),min(159.0,floor(q.x*160.0/maximum)));
    } else if(kind<4) {
        if(q.z>=0.0&&q.z<=maximum)cell=vec2(96*(kind-2)+min(95.0,floor(q.z*96.0/maximum)),0.0);
        amount*=q[kind-2];
    } else {
        float sector=clamp(floor((q.w+3.141592653589793)*(48.0/(2.0*3.141592653589793))),0.0,47.0);
        cell=vec2(192.0+sector,0.0);
        amount*=sqrt(max(q.x+q.y,0.0));
    }
    if(!(amount>0.0))cell=vec2(-1000.0);
    gl_Position=vec4(2.0*(cell+corner)/target_size-1.0,0.0,1.0);
}
''')
    info.fragment_source('void main(){fragColor=vec4(amount,0.0,0.0,1.0);}')
    _density_shader=gpu.shader.create_from_info(info)
    corners=np.array([(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)],np.float32)
    _density_batch=batch_for_shader(_density_shader,'TRIS',dict(corner=corners))
    return _density_shader,_density_batch


def _accumulate(signal,maximum,histogram=False):
    import gpu
    shader,batch=_density();width,height=(240,1) if histogram else (336,160)
    target=gpu.types.GPUTexture((width,height),format='R32F')
    fb=gpu.types.GPUFrameBuffer(color_slots=(target,));saved=_state()
    try:
        with fb.bind():
            _viewport(width,height);fb.clear(color=(0.,0.,0.,0.));gpu.state.blend_set('ADDITIVE')
            shader.bind();shader.uniform_sampler('states',signal['state']);shader.uniform_sampler('rgb_values',signal['rgb'])
            shader.uniform_sampler('meta_values',signal['meta']);shader.uniform_float('maximum',maximum)
            shader.uniform_int('source_width',signal['width'])
            shader.uniform_float('target_size',(width,height))
            for kind in ((2,3,4) if histogram else (0,1)):
                shader.uniform_int('kind',kind)
                if signal['count']:batch.draw_instanced(shader,instance_count=signal['count'])
    finally:_restore(saved)
    return target


def _reduction():
    global _row_shader,_max_shader,_quant_shader
    if _row_shader is not None:return
    import gpu
    def shader(source):
        info=gpu.types.GPUShaderCreateInfo();info.local_group_size(64,1,1)
        info.sampler(0,'FLOAT_2D','density');info.sampler(1,'FLOAT_2D','maxima')
        info.image(0,'R32F','FLOAT_2D','output_values',qualifiers={'WRITE'});info.compute_source(source)
        return gpu.shader.create_from_info(info)
    _row_shader=shader(r'''
void main(){int id=int(gl_GlobalInvocationID.x);if(id>=320)return;
int layer=id/160,row=id%160;float value=1.0;int width=layer==0?144:192,offset=layer==0?0:144;
if(layer==1||row<144)for(int x=0;x<192;x++){if(x>=width)break;value=max(value,texelFetch(density,ivec2(x+offset,row),0).r);}
imageStore(output_values,ivec2(row,layer),vec4(value));}
''')
    _max_shader=shader(r'''
void main(){int id=int(gl_GlobalInvocationID.x);if(id>=2)return;float value=1.0;
for(int row=0;row<160;row++)value=max(value,texelFetch(density,ivec2(row,id),0).r);
imageStore(output_values,ivec2(id,0),vec4(value));}
''')
    _quant_shader=shader(r'''
void main(){int id=int(gl_GlobalInvocationID.x);if(id>=336*160)return;ivec2 at=ivec2(id%336,id/336);
int layer=at.x<144?0:1;float value=max(texelFetch(density,at,0).r,0.0);
float peak=texelFetch(maxima,ivec2(layer,0),0).r;
float level=clamp(roundEven(log(1.0+value)/log(1.0+peak)*7.0),0.0,7.0);
imageStore(output_values,at,vec4(level/255.0));}
''')


def _levels(target):
    import gpu
    _reduction()
    rows=gpu.types.GPUTexture((160,2),format='R32F');maxima=gpu.types.GPUTexture((2,1),format='R32F')
    result=gpu.types.GPUTexture((336,160),format='R32F')
    for shader,source,peak,dest,count in ((_row_shader,target,target,rows,320),
        (_max_shader,rows,rows,maxima,2),(_quant_shader,target,maxima,result,336*160)):
        shader.bind();shader.uniform_sampler('density',source)
        if shader==_quant_shader:shader.uniform_sampler('maxima',peak)
        shader.image('output_values',dest);gpu.compute.dispatch(shader,(count+63)//64,1,1)
    fb=gpu.types.GPUFrameBuffer(color_slots=(result,))
    with fb.bind():
        values=np.array(fb.read_color(0,0,336,160,1,0,'UBYTE'),np.uint8).reshape(160,336)
    return dict(waveform_levels=np.ascontiguousarray(values[:,144:]),
                vectorscope_levels=np.ascontiguousarray(values[:144,:144]))


def jc(output,model,checkpoint=None):
    if checkpoint is not None:checkpoint()
    result=_levels(_accumulate(output,model.tone.plot_max*100))
    if output['comparison'] is not None:
        if checkpoint is not None:checkpoint()
        result['_comparison']=_levels(_accumulate(output['comparison'],model.tone.plot_max*100))
    return result


def histograms(selected,model,checkpoint=None):
    import gpu
    if checkpoint is not None:checkpoint()
    inputs=selected['gpu']
    if inputs['drivers'] is None:inputs['drivers']=_engine.evaluate(model,inputs,2)
    target=_accumulate(inputs['drivers'],model.tone.plot_max*100,True)
    # One cold readback contains both curve histograms and the polar histogram.
    values=np.array(target.read(),np.float32).reshape(240)
    values/=max(float(np.sum(selected['alpha']*selected.get('mass',1.))),1e-20)
    return dict(low=values[:96].copy(),high=values[96:192].copy(),sectors=values[192:].copy())


def gather(output,ids,checkpoint=None):
    global _gather_shader
    import gpu
    if checkpoint is not None:checkpoint()
    ids=np.asarray(ids,dtype=np.int32)
    if len(ids)>512:raise ValueError('Scope statistical sample exceeds 512 points')
    if not len(ids):return np.empty((0,2),np.float32)
    if _gather_shader is None:
        info=gpu.types.GPUShaderCreateInfo();info.local_group_size(64,1,1)
        info.sampler(0,'FLOAT_2D','states');info.sampler(1,'FLOAT_2D','indices')
        info.image(0,'RG32F','FLOAT_2D','points',qualifiers={'WRITE'})
        info.push_constant('INT','count')
        info.compute_source(r'''
void main(){int id=int(gl_GlobalInvocationID.x);if(id>=count)return;
int index=int(texelFetch(indices,ivec2(id,0),0).r);vec4 q=texelFetch(states,ivec2(index%256,index/256),0);
vec2 value=vec2(0.0);float root=sqrt(abs(q.w));
if(root>0.0)value=q.yz/root;
else if(any(notEqual(q.yz,vec2(0.0))))value=vec2(uintBitsToFloat(0x7fc00000u));
if(any(isnan(q))||any(isinf(q)))value=vec2(uintBitsToFloat(0x7fc00000u));
imageStore(points,ivec2(id,0),vec4(value,0.0,0.0));}
''')
        _gather_shader=gpu.shader.create_from_info(info)
    # `output` normally carries the selected group's helper-owned cache via
    # `evaluate`.  Keep this fallback local for conformance tests and other
    # direct callers that construct a minimal output dictionary.
    cache=output.get('_gather_cache')
    cached_ids=None if cache is None else cache.get('ids')
    if (cache is None or cached_ids is None or len(cached_ids)!=len(ids)
            or not np.array_equal(cached_ids,ids)):
        values=np.ascontiguousarray(ids,dtype=np.float32)
        indices=gpu.types.GPUTexture((len(ids),1),format='R32F',
            data=gpu.types.Buffer('FLOAT',(1,len(ids)),values))
        points=gpu.types.GPUTexture((len(ids),1),format='RG32F')
        if cache is not None:
            frozen=ids.copy()
            frozen.setflags(write=False)
            cache.update(ids=frozen,indices=indices,points=points)
    else:
        indices,points=cache['indices'],cache['points']
    shader=_gather_shader;shader.bind();shader.uniform_sampler('states',output['state'])
    shader.uniform_sampler('indices',indices);shader.uniform_int('count',len(ids));shader.image('points',points)
    gpu.compute.dispatch(shader,(len(ids)+63)//64,1,1)
    return np.array(points.read(),np.float32).reshape(-1,2)


def clear():
    global _engine,_density_shader,_density_batch,_row_shader,_max_shader,_quant_shader,_gather_shader
    _engine=None;_density_shader=None;_density_batch=None
    _row_shader=None;_max_shader=None;_quant_shader=None;_gather_shader=None
