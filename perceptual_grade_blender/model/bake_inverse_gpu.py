"""Actual Blender compute counterpart of bake_inverse; no CPU inverse fallback."""
import time
import numpy as np
from . import bake_gpu, boundary_gpu
from ...perceptual_grade_core.model import bake_inverse, tone_table


def source(data,offsets,expanded):
    constants=(f'const float PG_INV_TOL={float(bake_inverse.TOLERANCE):.9g};\n'
               f'const float PG_SEARCH_TOL={float(bake_inverse.SEARCH_TOLERANCE):.9g};\n'
               f'const float PG_CHROMA_TOE_MAX={float(bake_inverse.CHROMA_TOE_MAX):.9e};\n'
               f'const float PG_CHROMA_TOE_MIN={float(bake_inverse.CHROMA_TOE_MIN):.9e};\n'
               f'const float PG_RGB_REL_TOL={float(bake_inverse.RGB_RELATIVE_TOLERANCE):.9g};\n'
               f'const float PG_RGB_ABS_TOL={float(bake_inverse.RGB_ABSOLUTE_TOLERANCE):.9g};\n'
               f'const float PG_INV_STEP={float(bake_inverse.STEP):.9g};\n'
               f'const int PG_INV_ITERATIONS={bake_inverse.ITERATIONS};\n')
    table_fetch='''
float pg_tone_node(int node,int component) {
    int index=4*node+component;
    return texelFetch(grade_parameters,ivec2(index%256,1+index/256),0).x;
}
'''
    table_source=tone_table.GLSL
    for old,offset in ((117,0),(118,1),(119,2),(120,3),(121,4)):
        table_source=table_source.replace(f'pg_param({old})',f'pg_param(PG_P_TONE_TABLE+{offset})')
    return bake_gpu.source(data,offsets,expanded).split('void main()',1)[0]+constants+table_fetch+table_source+MATH


MATH=r'''
vec3 pg_canonical(vec3 v) {
    return pg_native(v,vec3(pg_param(PG_P_CANONICAL_FIRST),pg_param(PG_P_CANONICAL_FIRST+1),pg_param(PG_P_CANONICAL_FIRST+2)),vec3(pg_param(PG_P_CANONICAL_LAST),pg_param(PG_P_CANONICAL_LAST+1),pg_param(PG_P_CANONICAL_LAST+2)));
}
vec3 pg_y_weights() {return vec3(pg_param(PG_P_Y_WEIGHTS),pg_param(PG_P_Y_WEIGHTS+1),pg_param(PG_P_Y_WEIGHTS+2));}
float pg_max3(vec3 v) {return max(v.x,max(v.y,v.z));}
float pg_error(vec4 v) {return max(max(abs(v.x),abs(v.y)),max(abs(v.z),abs(v.w)));}
vec4 pg_features(vec4 q) {return vec4(q.x,sign(q.w)*sqrt(abs(q.w)),q.yz);}
vec4 pg_state(float ja,float c,float angle) {return vec4(sqrt(ja*ja+66.0*c),c*cos(angle),c*sin(angle),ja*ja);}
vec3 pg_input_repair(vec3 rgb) {
    if(all(greaterThanEqual(rgb,vec3(0.0)))&&all(lessThanEqual(rgb,vec3(1024.0))))return rgb;
    float amplitude=pg_max3(abs(rgb));vec3 unit=rgb/amplitude;
    float y=max(0.0,dot(unit,pg_y_weights())),k=max(0.0,-min(unit.x,min(unit.y,unit.z)));
    if(k>0.0)unit=(unit+k)*pg_div(y,y+k);
    float limit=pg_div(1024.0,pg_max3(unit));
    float down=limit>0.0?uintBitsToFloat(floatBitsToUint(limit)-1u):0.0;
    return unit*max(0.0,min(amplitude,down));
}
float pg_inverse_hypot(vec2 value) {
    float peak=max(abs(value.x),abs(value.y));
    if(peak==0.0)return 0.0;
    if(peak<1e-15||peak>1e15)return peak*length(value/peak);
    return length(value);
}
vec2 pg_chroma_scale(vec4 target) {
    float ceiling=66.0*PG_CHROMA_TOE_MAX;
    float h=min(abs(target.x),sqrt(ceiling));
    float energy=max(h*h,min(abs(target.w),ceiling));
    float c=pg_inverse_hypot(min(abs(target.yz),vec2(PG_CHROMA_TOE_MAX)));
    float tau=max(PG_CHROMA_TOE_MIN,min(PG_CHROMA_TOE_MAX,max(energy/66.0,c)));
    return vec2(tau,log(tau));
}
float pg_chroma_encode(float log_value,float log_scale) {
    float ell=log_value-log_scale;
    if(ell<=0.0) {float u=exp(ell);return u*(2.0-.5*u);}
    return 1.5+ell;
}
float pg_chroma_decode(float coordinate,float log_scale) {
    if(coordinate<=1.5)return exp(log_scale)*(2.0*coordinate/(2.0+sqrt(4.0-2.0*coordinate)));
    return exp(coordinate-1.5+log_scale);
}
float pg_chroma_residual(float radius,float c,float gain,vec2 scale) {
    float logr=radius>0.0?log(radius):0.0,logc=c>0.0?log(c):0.0;
    float logpredicted=logc+.6931471805599453*gain;
    if(radius>scale.x&&c>0.0&&logpredicted>scale.y)
        return logr-logc-.6931471805599453*gain;
    float actual=radius>0.0?pg_chroma_encode(logr,scale.y):0.0;
    float predicted=c>0.0?pg_chroma_encode(logpredicted,scale.y):0.0;
    return actual-predicted;
}
vec4 pg_undo_tone(vec4 q) {
    if(pg_param(0)==0.0)return q;
    float j=sqrt(max(q.w,0.0)),c=length(q.yz),h=sqrt(j*j+66.0*c);
    float original=h==0.0?0.0:100.0*(pg_param(1)+pg_param(2)*pg_tone_table_inverse(h));
    float gain=pg_div(2.0*original*original,pg_inverse_hypot(vec2(66.0*c,2.0*j*original))+66.0*c);
    float jj=j*gain;vec2 ab=q.yz*gain;
    return vec4(sqrt(jj*jj+66.0*length(ab)),ab,jj*jj);
}
vec4 pg_direct_state(vec4 raw) {
    vec4 q=raw;
    if(pg_param(PG_P_GAIN)!=0.0) {
        q.yz*=exp2(-pg_param(PG_P_GAIN));float c=length(q.yz);
        q.w=max(0.0,q.x*q.x-66.0*c);q.x=sqrt(q.w+66.0*c);
    }
    return pg_undo_tone(q);
}
vec2 pg_opponent_encode(vec2 ab,float logscale) {
    float c=pg_inverse_hypot(ab);
    float encoded=c>0.0?pg_chroma_encode(log(c),logscale):0.0;
    return ab*pg_div(encoded,c);
}
vec2 pg_opponent_decode(vec2 x,float logscale) {
    float radius=pg_inverse_hypot(x);
    return x*pg_div(pg_chroma_decode(radius,logscale),radius);
}
vec4 pg_reduced(vec3 x,vec4 target,vec2 scale,bool reconstruct,out vec4 state) {
    float j=exp(x.x);vec2 ab=pg_opponent_decode(x.yz,scale.y);
    vec4 candidate=vec4(sqrt(j*j+66.0*pg_inverse_hypot(ab)),ab,j*j);
    state=candidate;
    if(reconstruct)return vec4(0.0);
    vec4 actual=pg_grade(candidate);
    vec2 difference=pg_opponent_encode(actual.yz,scale.y)-pg_opponent_encode(target.yz,scale.y);
    float ja=sign(actual.w)*sqrt(abs(actual.w)),wanted=sign(target.w)*sqrt(abs(target.w));
    return vec4((actual.x-target.x)/(1.0+abs(target.x)),(ja-wanted)/(1.0+abs(wanted)),difference);
}

float pg_acceptance_limit(vec4 target,bool appearance) {
    if(appearance)return PG_INV_TOL;
    vec3 compared=pg_param(PG_P_IDENTITY)!=0.0?target.xyz:pg_canonical(target.xyz);
    float peak=pg_max3(abs(compared));
    return max(PG_RGB_REL_TOL*peak,PG_RGB_ABS_TOL)/max(peak,1e-4);
}
vec4 pg_inverse_output_residual(vec4 actual,vec4 reference,vec4 target,bool appearance) {
    if(appearance) {vec4 wanted=pg_features(target);return (pg_features(actual)-wanted)/(1.0+abs(wanted));}
    vec3 difference=pg_canonical(pg_export_reference(actual,reference)-target.xyz);
    return vec4(difference/max(pg_max3(abs(pg_canonical(target.xyz))),1e-4),0.0);
}
vec4 pg_inverse_residual(vec3 rgb,vec4 target,bool appearance) {
    if(pg_param(PG_P_IDENTITY)!=0.0&&!appearance)return vec4((rgb-target.xyz)/max(pg_max3(abs(target.xyz)),1e-4),0.0);
    vec4 toned=pg_tone(pg_enter(rgb));
    return pg_inverse_output_residual(pg_grade(toned),toned,target,appearance);
}
vec4 pg_fit_residual(vec3 x,vec4 target,vec2 scale) {
    vec4 state;return pg_reduced(x,target,scale,false,state);
}
vec3 pg_normal_step(vec4 j[3],vec4 residual,float damping) {
    float a=dot(j[0],j[0]),b=dot(j[1],j[1]),c=dot(j[2],j[2]);
    float ab=dot(j[0],j[1]),ac=dot(j[0],j[2]),bc=dot(j[1],j[2]);
    float lam=damping*(1.0+a+b+c);
    float l00=sqrt(a+lam),l10=ab/l00,l20=ac/l00;
    float l11=sqrt(max(b+lam-l10*l10,lam)),l21=(bc-l20*l10)/l11;
    float l22=sqrt(max(c+lam-l20*l20-l21*l21,lam));
    float g0=dot(j[0],residual),g1=dot(j[1],residual),g2=dot(j[2],residual);
    float y0=g0/l00,y1=(g1-l10*y0)/l11,y2=(g2-l20*y0-l21*y1)/l22;
    float z2=y2/l22,z1=(y1-l21*z2)/l11,z0=(y0-l10*z1-l20*z2)/l00;
    return vec3(z0,z1,z2);
}
vec3 pg_fit(vec3 seed,vec4 target,vec2 scale) {
    float bound=pg_chroma_encode(20.0,scale.y);
    vec3 low=vec3(-69.0,-bound,-bound),high=vec3(20.0,bound,bound);
    vec3 x=clamp(seed,low,high);vec4 r=pg_fit_residual(x,target,scale);
    float score=dot(r,r),damping=1e-4;bool fresh=true,measured=false;
    vec4 jac[3];for(int k=0;k<3;k++)jac[k]=vec4(0.0);
    for(int iteration=0;iteration<PG_INV_ITERATIONS;iteration++) {
        if(pg_error(r)<=PG_SEARCH_TOL)break;
        if(fresh||(iteration%12==0&&!measured)) {
            for(int k=0;k<3;k++) {
                vec3 delta=vec3(0.0);delta[k]=PG_INV_STEP;
                vec3 hi=clamp(x+delta,low,high),lo=clamp(x-delta,low,high);
                jac[k]=(pg_fit_residual(hi,target,scale)-pg_fit_residual(lo,target,scale))/(hi[k]-lo[k]);
            }
            measured=true;
        }
        vec3 step=pg_normal_step(jac,r,damping);
        step/=max(1.0,max(max(abs(step.x),abs(step.y)),abs(step.z))/1.5);
        bool improved=false;
        for(int trial=0;trial<3;trial++) {
            float alpha=trial==0?1.0:(trial==1?.5:.125);
            vec3 candidate=clamp(x-alpha*step,low,high);vec4 rr=pg_fit_residual(candidate,target,scale);
            float ss=dot(rr,rr);
            if(ss<score) {
                vec3 delta=candidate-x;vec4 change=rr-r-jac[0]*delta.x-jac[1]*delta.y-jac[2]*delta.z;
                float square=dot(delta,delta);vec3 direction=square>1e-20?delta/square:vec3(0.0);
                for(int k=0;k<3;k++)jac[k]+=change*direction[k];
                measured=false;x=candidate;r=rr;score=ss;improved=true;break;
            }
        }
        fresh=!improved&&!measured;
        damping=improved?max(1e-6,damping*.2):min(1e6,damping*10.0);
    }
    return x;
}

void pg_choose_error(vec3 candidate,float rr,float tolerance,vec3 prior,
                     inout vec3 best,inout float error,inout float distance,inout vec3 canonical_best) {
    vec3 canonical=pg_canonical(candidate);
    vec3 delta=asinh(canonical)-prior;float dd=dot(delta,delta);
    bool root=rr<=tolerance,oldroot=error<=tolerance;
    bool lex=canonical.x<canonical_best.x||(canonical.x==canonical_best.x&&
        (canonical.y<canonical_best.y||(canonical.y==canonical_best.y&&canonical.z<canonical_best.z)));
    bool near=dd<distance||(dd==distance&&lex);
    bool pick=(root&&!oldroot)||((root==oldroot)&&(root?near:(rr<error||(rr==error&&near))));
    if(pick) {best=candidate;error=rr;distance=dd;canonical_best=canonical;}
}
void pg_choose(vec3 candidate,vec4 target,bool appearance,float tolerance,vec3 prior,
               inout vec3 best,inout float error,inout float distance,inout vec3 canonical_best) {
    candidate=clamp(candidate,0.0,1024.0);
    float rr=pg_error(pg_inverse_residual(candidate,target,appearance));
    pg_choose_error(candidate,rr,tolerance,prior,best,error,distance,canonical_best);
}
vec4 pg_inverse(vec4 target,bool appearance) {
    if(!appearance)target.xyz=pg_input_repair(target.xyz);
    vec3 prior=appearance?clamp(pg_export(target),0.0,1024.0):target.xyz;
    if(pg_param(PG_P_IDENTITY)!=0.0)return vec4(prior,pg_error(pg_inverse_residual(prior,target,appearance)));
    vec3 best=prior,canonical_best=pg_canonical(prior),reference=asinh(canonical_best);
    vec4 entered=pg_enter(prior),toned=pg_tone(entered);
    float error=pg_error(pg_inverse_output_residual(pg_grade(toned),toned,target,appearance)),distance=0.0;
    float tolerance=pg_acceptance_limit(target,appearance);
    if(error<=tolerance)return vec4(best,error);
    float black_error=pg_error(pg_inverse_output_residual(vec4(0.0),vec4(0.0),target,appearance));
    pg_choose_error(vec3(0.0),black_error,tolerance,reference,best,error,distance,canonical_best);
    vec4 raw=appearance?target:entered;
    if(pg_param(PG_P_INVERSE_SIMPLE)!=0.0) {
        vec3 regular=clamp(pg_export(pg_direct_state(raw)),0.0,1024.0);
        pg_choose(regular,target,appearance,tolerance,reference,best,error,distance,canonical_best);
    } else {
        vec2 scale=pg_chroma_scale(raw);
        vec2 ab=toned.yz*exp2(-pg_param(PG_P_GAIN));
        vec3 seed=vec3(log(max(sqrt(max(toned.w,0.0)),1e-30)),pg_opponent_encode(ab,scale.y));
        vec3 fit=pg_fit(seed,raw,scale);vec4 q;pg_reduced(fit,raw,scale,true,q);
        pg_choose(pg_export(pg_undo_tone(q)),target,appearance,tolerance,reference,best,error,distance,canonical_best);
    }
    return vec4(best,error);
}
void main() {
    int id=int(gl_GlobalInvocationID.x);if(id>=sample_count)return;
    ivec2 address=ivec2(id%256,id/256);vec4 input_value=texelFetch(input_pixels,address,0),result;
    if(operation<2)result=pg_inverse(input_value,operation==1);
    else if(operation==7)result=vec4(pg_chroma_scale(input_value),0.0,0.0);
    else if(operation==6) {
        float logscale=log(input_value.y),encoded=0.0,scaled=0.0;
        if(input_value.x>0.0) {
            float logvalue=log(input_value.x);encoded=pg_chroma_encode(logvalue,logscale);
            scaled=pg_chroma_encode(logvalue+.6931471805599453*input_value.z,logscale);
        }
        result=vec4(encoded,pg_chroma_decode(input_value.w,logscale),scaled,
            pg_chroma_residual(input_value.x,input_value.x,input_value.z,vec2(input_value.y,logscale)));
    }
    else if(operation==5)result=pg_undo_tone(input_value);
    else if(operation==4)result=pg_enter(input_value.xyz);
    else if(operation==3)result=pg_grade(pg_tone(pg_enter(input_value.xyz)));
    else result=vec4(pg_forward(input_value.xyz),0.0);
    imageStore(output_pixels,address,result);
}
'''


class Program(boundary_gpu.Program):
    def __init__(self,inverse):
        k=inverse.kernel
        super().__init__(k.data,source,True,k.table.expanded)
        self.set_inverse(inverse)

    def evaluate(self,points,operation=0):
        if operation in (0,1,5) and self.missing_tone_table:
            raise RuntimeError('Inverse tone table must be prepared before compute evaluation')
        return super().evaluate(points,operation)

    def set_inverse(self,inverse):
        start=time.perf_counter()
        k=inverse.kernel;values=bake_gpu.parameters(k)
        layout=bake_gpu.LAYOUT
        def put(name,value):
            value=np.asarray(value,dtype='f4').reshape(-1);start=layout[name]
            values[start:start+len(value)]=value
        put('CANONICAL_FIRST',inverse.canonical[:,0]);put('CANONICAL_LAST',inverse.canonical[:,2])
        put('Y_WEIGHTS',inverse.weights);put('INVERSE_SIMPLE',inverse.simple)
        put('DIRECTIONAL',k.directional_drivers)
        table=inverse.tone_table
        self.missing_tone_table=k.tone.active and table is None
        self.tone_lookup_bytes=0 if table is None else table.bytes
        if table is not None:
            put('TONE_TABLE',[table.scale,table.tail_h,table.slope,table.intercept,len(table.nodes)])
            packed=np.zeros((1+(table.nodes.size+255)//256,256),dtype='f4')
            packed[0]=values;packed[1:].reshape(-1)[:table.nodes.size]=table.nodes.ravel()
        else:packed=values.reshape(1,256)
        self.parameter_bytes=packed.nbytes
        self.parameters=self.gpu.types.GPUTexture((256,len(packed)),format='R32F',
            data=self.gpu.types.Buffer('FLOAT',packed.shape,packed))
        self.parameter_upload_seconds=time.perf_counter()-start
        self.parameter_upload_total_seconds=getattr(self,'parameter_upload_total_seconds',0.)+self.parameter_upload_seconds
