"""Blender GLSL uint16 faces and polynomial crease/profile/guard lookup."""
import time
import numpy as np
from ...perceptual_grade_core.model.boundary_polynomial import expand

FIELDS=('edge_axis','edge_polynomial','vertex_t','edges','heights','topology','edge_hints',
        'face_low','face_span','face_offset','face_scale','guard_axis','guard_polynomial',
        'profile_axis','profile_polynomial')
WIDTH=256
BATCH=4096


def pack(data,expanded=None):
    expanded=expand(data) if expanded is None else expanded
    arrays=[];offsets={};count=0
    for name in FIELDS:
        if name not in expanded:continue
        values=np.asarray(expanded[name],dtype=np.float32)
        if name=='edge_polynomial':values=values.transpose(0,1,3,2)
        values=values.ravel()
        offsets[name]=count;count+=len(values);arrays.append(values)
        padding=(-count)%4
        if padding:arrays.append(np.zeros(padding,dtype=np.float32));count+=padding
    values=np.concatenate(arrays)
    padded=np.zeros((len(values)+4*WIDTH-1)//(4*WIDTH)*4*WIDTH,dtype=np.float32)
    padded[:len(values)]=values
    return padded.reshape(-1,WIDTH,4),offsets


def header(data,offsets,expanded):
    constants='\n'.join(f'const int PG_{name.upper()}={value};' for name,value in offsets.items())
    counts=dict(EDGE=len(expanded['edge_axis']),FACE=expanded['face_codes'].shape[1],
                GUARD=len(expanded.get('guard_axis',[])),PROFILE=len(expanded.get('profile_axis',[])),
                HINT_T=expanded['edge_hints'].shape[1],HINT_H=expanded['edge_hints'].shape[2])
    constants+='\n'+'\n'.join(f'const int PG_{name}_COUNT={value};' for name,value in counts.items())
    constants+=f"\nconst float PG_WHITE={float(np.float32(data['white'])):.9g};\nconst float PG_GAMMA={float(np.float32(.525*(1.48+np.sqrt(.1)))):.9g};\n"
    math=MATH
    if 'guard_axis' not in expanded:
        math=math[:math.index('float pg_profile(float t)')]
    return constants+math


def source(data,offsets,expanded):
    return header(data,offsets,expanded)+'''\nvoid main() {
    int id=int(gl_GlobalInvocationID.x);
    if(id>=sample_count)return;
    ivec2 address=ivec2(id%256,id/256);
    vec4 p=texelFetch(input_pixels,address,0);
    int face;
    float unit=pg_capacity(p.x,p.y,face);
    vec2 guard=pg_intersection(p.z);
    imageStore(output_pixels,address,vec4(unit,float(face),guard));
}
'''


MATH=r'''
float pg_face_code(int index) {
    return round(texelFetch(face_atlas,ivec2(index%PG_FACE_COUNT,index/PG_FACE_COUNT),0).r*65535.0);
}
float pg_fetch(int index) {
    int texel=index/4;
    return texelFetch(boundary_cache,ivec2(texel%256,texel/256),0)[index%4];
}
vec4 pg_coefficients(int index) {
    int texel=index/4;
    return texelFetch(boundary_cache,ivec2(texel%256,texel/256),0);
}
float pg_horner(vec4 c,float z) {
    return ((c.w*z+c.z)*z+c.y)*z+c.x;
}
float pg_mod(float h) {
    const float tau=6.283185307179586;
    if(h>= -2.0*tau&&h<2.0*tau)
        return h+(h< -tau?2.0*tau:(h<0.0?tau:(h>=tau?-tau:0.0)));
    // Float32 tau = 13176795 * 2^-21. Reduce the significand using
    // integers: GLSL mod loses the remainder for large finite floats.
    uint bits=floatBitsToUint(h)&0x7fffffffu;
    int shift=int(bits>>23u)-129;
    uint r=(bits&0x7fffffu)|0x800000u;
    if(r>=13176795u)r-=13176795u;
    for(int i=0;i<125;i++) {
        if(i>=shift)break;
        r+=r;if(r>=13176795u)r-=13176795u;
    }
    float result=float(r)*(1.0/2097152.0);
    return h<0.0&&r!=0u?tau-result:result;
}
void pg_address(int offset,int count,float x,out int index,out float fraction) {
    int lo=0,hi=count;
    // Fixed binary addressing, not an equation solve.
    for(int step=0;step<9;step++) {
        if(lo<hi) {
            int mid=(lo+hi)/2;
            if(pg_fetch(offset+mid)<=x)lo=mid+1;else hi=mid;
        }
    }
    index=clamp(lo-1,0,count-2);
    float left=pg_fetch(offset+index),right=pg_fetch(offset+index+1);
    fraction=(x-left)/(right-left);
}
float pg_curve(int axis,int count,int coefficients,float x) {
    int i;float z;pg_address(axis,count,x,i,z);int at=coefficients+4*i;
    return pg_horner(pg_coefficients(at),z);
}
float pg_vertex(int id) { return pg_fetch(PG_VERTEX_T+id); }
void pg_edge_address(int edge,float t,out int base,out float z) {
    float low=pg_vertex(int(pg_fetch(PG_EDGES+2*edge)));
    float high=pg_vertex(int(pg_fetch(PG_EDGES+2*edge+1)));
    float x=clamp((t-low)/(high-low),0.0,1.0);
    int i=clamp(int(x*float(PG_EDGE_COUNT-1)),0,PG_EDGE_COUNT-2);
    if(x<pg_fetch(PG_EDGE_AXIS+i))i=max(0,i-1);
    if(i<PG_EDGE_COUNT-2&&x>=pg_fetch(PG_EDGE_AXIS+i+1))i++;
    float a=pg_fetch(PG_EDGE_AXIS+i),b=pg_fetch(PG_EDGE_AXIS+i+1);z=(x-a)/(b-a);
    base=PG_EDGE_POLYNOMIAL+(edge*(PG_EDGE_COUNT-1)+i)*8;
}
float pg_edge_value(int base,float z,int channel) {
    return pg_horner(pg_coefficients(base+4*channel),z);
}
float pg_capacity(float t,float hue,out int face) {
    int region=t==0.0?0:1;
    for(int i=1;i<8;i++) {
        float height=pg_fetch(PG_HEIGHTS+i);
        if(t>=height)region=2*i+(t>height?1:0);
    }
    int mask=31;
    if(region%2==1&&abs(hue)<=25.132741228718345) {
        int band=region/2;
        float low=pg_fetch(PG_HEIGHTS+band),high=pg_fetch(PG_HEIGHTS+band+1);
        int i=clamp(int((t-low)/(high-low)*float(PG_HINT_T_COUNT)),0,PG_HINT_T_COUNT-1);
        int j=min(int(pg_mod(hue)/6.283185307179586*float(PG_HINT_H_COUNT)),PG_HINT_H_COUNT-1);
        mask=int(pg_fetch(PG_EDGE_HINTS+(band*PG_HINT_T_COUNT+i)*PG_HINT_H_COUNT+j));
    }
    ivec4 chosen=ivec4(0);float dl=1e30,left_z=0.0;int left_base=0;
    for(int k=0;k<5;k++) {
        if((mask&(1<<k))==0)continue;
        ivec4 candidate=ivec4(pg_coefficients(PG_TOPOLOGY+(region*5+k)*4));
        int base;float z;pg_edge_address(candidate.x,t,base,z);
        float angle=pg_mod(pg_edge_value(base,z,0));
        float distance=pg_mod(hue-angle);
        if(distance<dl) {dl=distance;chosen=candidate;left_base=base;left_z=z;}
    }
    int right=chosen.y,right_base;float right_z;face=chosen.w;
    pg_edge_address(right,t,right_base,right_z);
    float dr=pg_mod(pg_mod(pg_edge_value(right_base,right_z,0))-hue);
    if(chosen.z!=right) {
        int base;float z;pg_edge_address(chosen.z,t,base,z);
        float other=pg_mod(pg_mod(pg_edge_value(base,z,0))-hue);
        if(other<dr) {dr=other;right_base=base;right_z=z;}
    }
    float left_radius=pg_edge_value(left_base,left_z,1),right_radius=pg_edge_value(right_base,right_z,1);
    float s=dl+dr>0.0?dl/(dl+dr):0.0;
    float u=(t-pg_fetch(PG_FACE_LOW+face))/pg_fetch(PG_FACE_SPAN+face)*float(PG_FACE_COUNT-1);
    float v=s*float(PG_FACE_COUNT-1);
    int i=clamp(int(u),0,PG_FACE_COUNT-2),j=clamp(int(v),0,PG_FACE_COUNT-2);
    float a=u-float(i),b=v-float(j);int index=(face*PG_FACE_COUNT+i)*PG_FACE_COUNT+j;
    float q00=pg_face_code(index),q01=pg_face_code(index+1);
    float row0=q00+a*(pg_face_code(index+PG_FACE_COUNT)-q00);
    float row1=q01+a*(pg_face_code(index+PG_FACE_COUNT+1)-q01);
    float correction=(row0+b*(row1-row0))*pg_fetch(PG_FACE_SCALE+face)+pg_fetch(PG_FACE_OFFSET+face);
    float reciprocal=(1.0-s)*left_radius+s*right_radius+s*(1.0-s)*correction;
    return 1.0/reciprocal;
}
float pg_profile(float t) {
    return pg_curve(PG_PROFILE_AXIS,PG_PROFILE_COUNT,PG_PROFILE_POLYNOMIAL,t);
}
vec2 pg_intersection(float h,out float response) {
    if(h<=0.0) {response=0.0;return vec2(0.0);}
    if(h>=PG_WHITE) {response=1.0;return vec2(0.0,PG_WHITE*PG_WHITE);}
    float x=h/PG_WHITE;
    float factor=pg_curve(PG_GUARD_AXIS,PG_GUARD_COUNT,PG_GUARD_POLYNOMIAL,x);
    float t=x*x*factor;
    float ja=min(h,PG_WHITE*pow(min(t,1.0),PG_GAMMA));
    response=min(t,1.0);
    return vec2((h-ja)*(h+ja)/66.0,ja*ja);
}
vec2 pg_intersection(float h) {float response;return pg_intersection(h,response);}
'''


class Program:
    """Small actual compute harness; callers initialize Blender's GPU context."""
    def __init__(self,data,source_builder=source,with_parameters=False,expanded=None):
        import gpu
        self.gpu=gpu
        expanded=expand(data) if expanded is None else expanded
        values,offsets=pack(data,expanded)
        codes=expanded['face_codes'];size=codes.shape[-1]
        # Blender accepts FLOAT upload buffers. R16 stores normalized uint16;
        # rounding a fetched value times 65535 recovers the original code.
        atlas=codes.reshape(-1,size).astype(np.float32)/np.float32(65535)
        self.lookup_bytes=values.nbytes+codes.nbytes
        if self.lookup_bytes>1024*1024:raise MemoryError('Boundary lookup exceeds 1 MiB')
        self.cache=gpu.types.GPUTexture((WIDTH,len(values)),format='RGBA32F',
            data=gpu.types.Buffer('FLOAT',values.shape,values))
        self.face_atlas=gpu.types.GPUTexture((size,len(atlas)),format='R16',
            data=gpu.types.Buffer('FLOAT',atlas.shape,atlas))
        info=gpu.types.GPUShaderCreateInfo();info.local_group_size(64,1,1)
        info.sampler(0,'FLOAT_2D','input_pixels');info.sampler(1,'FLOAT_2D','boundary_cache')
        info.sampler(3,'FLOAT_2D','face_atlas')
        info.image(0,'RGBA32F','FLOAT_2D','output_pixels',qualifiers={'WRITE'})
        self.with_parameters=with_parameters
        if with_parameters:
            info.sampler(2,'FLOAT_2D','grade_parameters')
            info.push_constant('INT','operation')
        info.push_constant('INT','sample_count');info.compute_source(source_builder(data,offsets,expanded))
        start=time.perf_counter();self.shader=gpu.shader.create_from_info(info)
        self.compile_seconds=time.perf_counter()-start
        self.output=gpu.types.GPUTexture((WIDTH,BATCH//WIDTH),format='RGBA32F')

    def evaluate(self,points,operation=2):
        gpu=self.gpu;points=np.asarray(points,dtype=np.float32);points=points.reshape(-1,points.shape[-1]);output=[]
        for start in range(0,len(points),BATCH):
            part=points[start:start+BATCH]
            values=np.zeros((BATCH//WIDTH,WIDTH,4),dtype=np.float32)
            values.reshape(-1,4)[:len(part),:part.shape[-1]]=part
            image=gpu.types.GPUTexture((WIDTH,BATCH//WIDTH),format='RGBA32F',
                data=gpu.types.Buffer('FLOAT',values.shape,values))
            self.shader.bind();self.shader.uniform_sampler('input_pixels',image)
            self.shader.uniform_sampler('boundary_cache',self.cache)
            self.shader.uniform_sampler('face_atlas',self.face_atlas)
            if self.with_parameters:
                self.shader.uniform_sampler('grade_parameters',self.parameters)
                self.shader.uniform_int('operation',operation)
            self.shader.image('output_pixels',self.output);self.shader.uniform_int('sample_count',len(part))
            gpu.compute.dispatch(self.shader,(len(part)+63)//64,1,1)
            output.append(np.array(self.output.read(),dtype=np.float32).reshape(-1,4)[:len(part)].copy())
        return np.concatenate(output) if output else np.empty((0,4),dtype=np.float32)
