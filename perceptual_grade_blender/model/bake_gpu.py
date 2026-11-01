"""Regular-Grade compute backend using the same float32 equations as bake.py."""
import numpy as np
from . import boundary_gpu
from ...perceptual_grade_core.model.bake import CHROMA_CROSSOVER


# One 256-float header shared by forward/inverse shaders. Following rows
# contain only the inverse tone lookup; they never overlap grading controls.
LAYOUT=dict(TONE_ACTIVE=0,BLACK=1,SPAN=2,SLOPE=3,HIGH=4,TONE_WIDTH=5,
    TONE_KNOTS=6,TONE_RESIDUAL=11,COEFF_KNOTS=15,FIT_WIDTHS=20,GAIN=22,HALF_SATURATION=23,
    LIGHT_ACTIVE=24,CHROMA_ACTIVE=25,ADAPTANCE_ACTIVE=26,WEB_ACTIVE=27,IDENTITY=28,
    HK_GUARD=29,DIRECTIONAL=30,INVERSE_SIMPLE=31,LIGHT=32,CHROMA=48,
    TONE_NORMALIZERS=72,TONE_CONSTANTS=76,COEFF_NORMALIZERS=80,COEFF_CONSTANTS=88,
    ANCHORS=96,LOG_RADII=100,ANGLES=124,LOG_CHROMA=148,HUE_RADIANS=172,LOG_SCALE=196,
    CANONICAL_FIRST=220,CANONICAL_LAST=223,Y_WEIGHTS=226,TONE_TABLE=229,ADAPTANCE_LOG_GAINS=234)
HEADER_SIZE=256


def parameters(kernel):
    tone,p=kernel.tone,kernel.color
    result=np.zeros(HEADER_SIZE,dtype=np.float32)
    def put(name,values):
        values=np.asarray(values,dtype='f4').reshape(-1);start=LAYOUT[name]
        result[start:start+len(values)]=values
    put('TONE_ACTIVE',[tone.active,tone.black,tone.span,tone.slope,tone.high,float(tone.width)])
    put('TONE_KNOTS',tone.knots);put('TONE_RESIDUAL',tone.residual)
    put('COEFF_KNOTS',tone.coefficient_knots)
    fits=(p.lightness_fit,p.chroma_fit)
    put('FIT_WIDTHS',[.05*2**(-fit) for fit in fits]);put('GAIN',p.global_gain);put('HALF_SATURATION',kernel.adaptance_half_saturation)
    put('LIGHT_ACTIVE',[kernel.active_controls[n] for n in ('lightness','chroma','adaptance')])
    put('WEB_ACTIVE',kernel.web_active);put('IDENTITY',kernel.identity);put('HK_GUARD',kernel.hk_guard)
    put('DIRECTIONAL',kernel.directional_drivers)
    for name,key in (('LIGHT','lightness'),('CHROMA','chroma')):put(name,kernel.controls[key])
    put('ADAPTANCE_LOG_GAINS',kernel.adaptance_log_gains)
    put('TONE_NORMALIZERS',kernel.tone_curve.normalizers);put('TONE_CONSTANTS',kernel.tone_curve.constants)
    for group,fit in enumerate(fits):
        curve=kernel.coefficient_curves[fit]
        result[LAYOUT['COEFF_NORMALIZERS']+4*group:LAYOUT['COEFF_NORMALIZERS']+4*group+4]=curve.normalizers
        result[LAYOUT['COEFF_CONSTANTS']+4*group:LAYOUT['COEFF_CONSTANTS']+4*group+4]=curve.constants
    put('ANCHORS',kernel.anchors);put('LOG_RADII',kernel.web_radii);put('ANGLES',kernel.web_angles)
    put('LOG_CHROMA',kernel.web_controls['log_chroma']);put('HUE_RADIANS',kernel.web_controls['hue_degrees']*np.float32(np.pi/180))
    put('LOG_SCALE',kernel.web_controls['log_scale'])
    return result


def source(data,offsets,expanded):
    matrix=np.asarray(data['matrices'][0],dtype=np.float32)
    inverse=np.linalg.inv(np.asarray(data['matrices'][0],dtype=np.float64)).astype('f4')
    scalar=lambda value:format(float(np.float32(value)),'.9g')+('' if '.' in format(float(np.float32(value)),'.9g') or 'e' in format(float(np.float32(value)),'.9g') else '.0')
    vec=lambda values:'vec3('+','.join(map(scalar,values))+')'
    constants=''.join(f'const int PG_P_{name}={index};\n' for name,index in LAYOUT.items())
    for name,m in [('TO_RGB',matrix),('TO_CONES',inverse)]:
        constants+=f'const vec3 PG_{name}_0={vec(m[:,0])};\nconst vec3 PG_{name}_2={vec(m[:,2])};\n'
    p=np.power(np.float32(data['fl'])*1024,np.float32(.42));rw=np.float32(400*p/(27.13+p))
    for name,value in [('FL',data['fl']),('AW',data['aw']),('RW',rw),('CHROMA_CROSSOVER',CHROMA_CROSSOVER)]:
        constants+=f'const float PG_{name}={scalar(value)};\n'
    effective=np.column_stack((matrix[:,0],1-matrix[:,0]-matrix[:,2],matrix[:,2])).astype(float)
    inverse_effective=np.linalg.inv(effective)
    limits=np.stack((1024*np.minimum(inverse_effective,0).sum(axis=1),1024*np.maximum(inverse_effective,0).sum(axis=1)))
    pp=np.power(abs(limits)*float(np.float32(data['fl'])),.42)
    limits=np.sign(limits)*400*pp/(27.13+pp)
    constants+=f'const vec3 PG_RESPONSE_LOW={vec(limits[0])};\nconst vec3 PG_RESPONSE_HIGH={vec(limits[1])};\n'
    return boundary_gpu.header(data,offsets,expanded)+constants+MATH


MATH=r'''
#ifdef GPU_METAL
// GLSL's precise qualifier is a namespace name in MSL. Keep arithmetic
// controls local to pg_power; these pragmas must work in Blender's MSL 2.2
// compile mode, or compilation fails and the runtime retains CPU fallback.
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wunknown-pragmas"
#pragma clang diagnostic error "-Wignored-pragmas"
#define PG_PRECISE
#else
#define PG_PRECISE precise
#endif
float pg_param(int i) {return texelFetch(grade_parameters,ivec2(i,0),0).x;}
float pg_div(float a,float b) {return b!=0.0?a/b:0.0;}
// Native atan/pow error can move a pixel across a cube-face crease. Reduce
// their arguments before polynomial evaluation so lookup coordinates retain
// float32 accuracy even on software drivers with coarse native intrinsics.
float pg_atan2(float y,float x) {
    float ax=abs(x),ay=abs(y),peak=max(ax,ay);
    if(peak==0.0)return 0.0;
    float r=min(ax,ay)/peak,z=r/(1.0+sqrt(1.0+r*r)),q=z*z;
    float polynomial=1.0+q*(-1.0/3.0+q*(1.0/5.0+q*(-1.0/7.0+q*(1.0/9.0+
        q*(-1.0/11.0+q*(1.0/13.0+q*(-1.0/15.0+q/17.0)))))));
    float angle=2.0*z*polynomial;
    if(ay>ax)angle=1.5707963267948966-angle;
    if(x<0.0)angle=3.141592653589793-angle;
    return (floatBitsToUint(y)&0x80000000u)!=0u?-angle:angle;
}
float pg_power(float x,float exponent) {
#ifdef GPU_METAL
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#endif
    // Callers supply a nonnegative base and positive exponent. The reduced
    // log argument is at most tanh(log(2)/4), and |u| <= log(2)/2.
    if(x==0.0)return 0.0;
    int e;float m=frexp(x,e);
    if(m<.7071067811865476) {m*=2.0;e--;}
    float y=(m-1.0)/(m+1.0),z=y*y;
    float logarithm=2.0*y*(1.0+z*(1.0/3.0+z*(1.0/5.0+z*(1.0/7.0+
        z*(1.0/9.0+z*(1.0/11.0+z/13.0))))));
    // Split the exponent so both integer products are exact float32 values;
    // this also works on drivers that implement fma as multiply then add.
    float leading=uintBitsToFloat(floatBitsToUint(exponent)&0xfffff000u);
    PG_PRECISE float product=float(e)*leading,remainder=float(e)*(exponent-leading);
    int n=int(round(product+remainder+exponent*logarithm*1.4426950408889634));
    PG_PRECISE float u=((product-float(n))*.6931471805599453+remainder*.6931471805599453)+exponent*logarithm;
    float result=1.0+u*(1.0+u*(.5+u*(1.0/6.0+u*(1.0/24.0+u*(1.0/120.0+
        u*(1.0/720.0+u*(1.0/5040.0+u/40320.0)))))));
    return ldexp(result,n);
}
#ifdef GPU_METAL
#pragma clang diagnostic pop
#endif
vec3 pg_power(vec3 x,vec3 exponent) {
    return vec3(pg_power(x.x,exponent.x),pg_power(x.y,exponent.y),pg_power(x.z,exponent.z));
}
float pg_expm1(float x) {
    if(abs(x)<.02)return x*(1.0+x*(.5+x*(1.0/6.0+x*(1.0/24.0+x/120.0))));
    return exp(x)-1.0;
}
float pg_log1p(float x) {
    if(abs(x)<.05)return x*(1.0+x*(-.5+x*(1.0/3.0+x*(-.25+x*(.2+x*(-1.0/6.0+x/7.0))))));
    return log(1.0+x);
}
float pg_softplus(float x) {return max(x,0.0)+pg_log1p(exp(-abs(x)));}
vec4 pg_kernels(float t,int knots,float width,int normalizer,int prepared) {
    if(t==0.0)return vec4(0.0);
    float tail=log(-pg_expm1(-t/width));vec4 result;
    for(int i=0;i<4;i++)result[i]=width*pg_softplus(pg_param(prepared+i)-
        pg_softplus((pg_param(knots+i+1)-t)/width)+tail)/pg_param(normalizer+i);
    return result;
}
vec4 pg_weights(vec4 u,bool adapt) {
    u=clamp(u,0.0,1.0);u.y=min(u.x,u.y);u.z=min(u.y,u.z);u.w=min(u.z,u.w);
    return vec4((adapt?u.x:1.0)-u.y,u.y-u.z,u.z-u.w,u.w);
}
float pg_eccentricity(float c,float s) {
    float c2=c*c-s*s,s2=2.0*c*s,c3=c2*c-s2*s,s3=s2*c+c2*s;
    float c4=c2*c2-s2*s2,s4=2.0*c2*s2;
    return 1.0-.0582*c-.0258*c2-.1347*c3+.0289*c4-.1475*s-.0308*s2+.0385*s3+.0096*s4;
}
vec3 pg_native(vec3 v,vec3 first,vec3 last) {return v.y+(v.x-v.y)*first+(v.z-v.y)*last;}
float pg_hk(float ja,float c) {
    float base=sqrt(66.0*c),magnitude=length(vec2(ja,base));
    return ja>=0.0?magnitude:2.0*base-magnitude;
}
float pg_physical_capacity(float energy,vec2 ab) {
    if(energy<=0.0||energy>=PG_WHITE*PG_WHITE)return 0.0;
    float t=pg_power(sqrt(energy)/PG_WHITE,1.0/PG_GAMMA);int face;
    float angle=length(ab)>0.0?pg_atan2(ab.y,ab.x):0.0;
    return PG_RW*t*(1.0-t)*pg_capacity(t,angle,face);
}
vec4 pg_enter(vec3 rgb) {
    vec3 cones=pg_native(rgb,PG_TO_CONES_0,PG_TO_CONES_2);
    vec3 p=pg_power(abs(cones)*PG_FL,vec3(.42)),r=sign(cones)*400.0*p/(27.13+p);
    float t=(2.0*r.x+r.y+.05*r.z)/(3.05*PG_RW);
    vec2 ab=vec2((r.x-r.y)+(r.z-r.y)/11.0,((r.x-r.z)+(r.y-r.z))/9.0);
    float radius=length(ab);vec2 direction=radius>0.0?ab/radius:vec2(0.0);
    float c=1204.0*pg_eccentricity(direction.x,direction.y)*radius/PG_AW;
    float ja=sign(t)*PG_WHITE*pg_power(abs(t),PG_GAMMA);
    return vec4(pg_hk(ja,c),c*direction,ja*abs(ja));
}
vec4 pg_tone(vec4 q) {
    if(pg_param(0)==0.0)return q;
    float ja=sign(q.w)*sqrt(abs(q.w)),c=length(q.yz),magnitude=length(vec2(ja,sqrt(66.0*c)));
    float t=max(0.0,(magnitude/100.0-pg_param(1))/pg_param(2));
    vec4 k=pg_kernels(t,PG_P_TONE_KNOTS,pg_param(PG_P_TONE_WIDTH),PG_P_TONE_NORMALIZERS,PG_P_TONE_CONSTANTS),residual;
    for(int i=0;i<4;i++)residual[i]=pg_param(11+i);
    float h=100.0*(pg_param(3)*t+dot(k,residual));if(t==1.0)h=pg_param(4);
    float gain=pg_div(2.0*h*h,length(vec2(66.0*c,2.0*ja*h))+66.0*c);
    float j=ja*gain;vec2 ab=q.yz*gain;
    return vec4(pg_hk(j,length(ab)),ab,j*abs(j));
}
float pg_smoother(float t) {t=clamp(t,0.0,1.0);return t*t*t*(10.0+t*(-15.0+6.0*t));}
void pg_cardinal(float h,out int row,out float weight) {
    row=0;for(int i=1;i<3;i++)if(h>=pg_param(PG_P_ANCHORS+i))row=i;
    weight=pg_smoother((h-pg_param(PG_P_ANCHORS+row))/(pg_param(PG_P_ANCHORS+row+1)-pg_param(PG_P_ANCHORS+row)));
}
float pg_blend_control(int base,int row,int column,int stride,float w) {
    float a=pg_param(base+row*stride+column),b=pg_param(base+(row+1)*stride+column);
    return a+(b-a)*w;
}
float pg_response_transform(float r,float gain) {
    if(gain==1.0)return r;
    float t=min(abs(r)/400.0,1.0);
    if(abs(r)<=400.0)return r*(gain/((1.0-t)+gain*t));
    float root=sqrt(abs(r)/400.0-1.0);
    float u=root*(2.0*root/(length(vec2(1.0,2.0*root))+1.0)),z=u/gain;
    return sign(r)*400.0*(1.0+z+z*z);
}
vec3 pg_response_transform(vec3 r,vec3 gain) {
    return vec3(pg_response_transform(r.x,gain.x),pg_response_transform(r.y,gain.y),pg_response_transform(r.z,gain.z));
}
float pg_response_delta(float r,float log_gain) {
    float m=exp(log_gain),t=min(abs(r)/400.0,1.0);
    if(abs(r)<=400.0)return r*pg_expm1(log_gain)*(1.0-t)/((1.0-t)+m*t);
    float root=sqrt(abs(r)/400.0-1.0);
    float u=root*(2.0*root/(length(vec2(1.0,2.0*root))+1.0)),z=u/m;
    return sign(r)*400.0*u*pg_expm1(-log_gain)*(1.0+u+z);
}
float pg_reach_weight(float chroma,float energy) {
    float a=pg_param(PG_P_HALF_SATURATION)*sqrt(abs(energy)),scale=max(a,chroma);
    if(scale==0.0)return 1.0;
    float x=a/scale,y=chroma/scale;x*=x;y*=y;
    return x*x/(x*x+y*y);
}
vec3 pg_to_responses(vec3 q,float energy) {
    float c=length(q.yz);vec2 co=c>0.0?q.yz/c:vec2(0.0);
    vec3 axis=PG_AW/(1204.0*pg_eccentricity(co.x,co.y))*(
        co.x*vec3(451.0,-891.0,-220.0)/1403.0+co.y*vec3(288.0,-261.0,-6300.0)/1403.0);
    float neutral=PG_AW/3.05*sign(energy)*pg_power(sqrt(abs(energy))/100.0,1.0/PG_GAMMA);
    return neutral+c*axis;
}
vec3 pg_from_components(float a,vec2 ab,out float energy) {
    float radius=length(ab);vec2 direction=radius>0.0?ab/radius:vec2(0.0);
    float c=1204.0*pg_eccentricity(direction.x,direction.y)*radius/PG_AW;
    float ja=100.0*sign(a)*pg_power(abs(a)/PG_AW,PG_GAMMA);
    energy=ja*abs(ja);
    return vec3(pg_hk(ja,c),c*direction);
}
void pg_adaptance(inout vec3 result,inout float energy,float height,float source_chroma,float source_energy) {
    int row;float w;pg_cardinal(height,row,w);
    vec3 logs;for(int i=0;i<3;i++)logs[i]=pg_blend_control(PG_P_ADAPTANCE_LOG_GAINS,row,i,3,w);
    vec3 m=exp(logs);
    float r0=PG_AW/3.05*pg_power(height/100.0,1.0/PG_GAMMA),t=min(r0/400.0,1.0);
    vec3 ratios=m/((1.0-t)+m*t);
    if(r0>400.0)ratios=pg_response_transform(vec3(r0),m)/r0;
    float k=3.05/((2.0*ratios.x+ratios.y)+.05*ratios.z),v=pg_reach_weight(source_chroma,source_energy);
    vec3 gain=exp(v*logs);float factor=exp(v*log(k));
    if(v==0.0||(all(equal(gain,vec3(1.0)))&&factor==1.0))return;
    bool neutral=source_chroma==0.0&&all(equal(result.yz,vec2(0.0)));
    float original_energy=energy;
    vec3 response=pg_to_responses(result,energy),delta;
    for(int i=0;i<3;i++)delta[i]=pg_response_delta(response[i],v*logs[i]);
    float c=length(result.yz);vec2 co=c>0.0?result.yz/c:vec2(0.0);
    vec2 ab=result.yz*(PG_AW/(1204.0*pg_eccentricity(co.x,co.y)));
    ab+=vec2((delta.x-delta.y)+(delta.z-delta.y)/11.0,((delta.x-delta.z)+(delta.y-delta.z))/9.0);
    float a=PG_AW*sign(energy)*pg_power(sqrt(abs(energy))/100.0,1.0/PG_GAMMA);
    a=(a+(2.0*delta.x+delta.y))+.05*delta.z;
    result=pg_from_components(a*factor,ab*factor,energy);
    if(neutral)energy=original_energy;
    result.x=pg_hk(sign(energy)*sqrt(abs(energy)),length(result.yz));
}
void pg_terms_height(float magnitude,out vec4 light,out vec4 chroma) {
    float t=pg_smoother(magnitude/pg_param(PG_P_HIGH));
    vec4 cumulative[2],terms[2];terms[0]=vec4(0.0);terms[1]=vec4(0.0);
    for(int group=0;group<2;group++) {
        if(pg_param(PG_P_LIGHT_ACTIVE+group)==0.0)continue;
        if(group==1&&pg_param(PG_P_LIGHT_ACTIVE)!=0.0&&pg_param(PG_P_FIT_WIDTHS)==pg_param(PG_P_FIT_WIDTHS+1))cumulative[group]=cumulative[0];
        else cumulative[group]=pg_kernels(t,PG_P_COEFF_KNOTS,pg_param(PG_P_FIT_WIDTHS+group),PG_P_COEFF_NORMALIZERS+4*group,PG_P_COEFF_CONSTANTS+4*group);
        vec4 w=pg_weights(cumulative[group],false);
        for(int i=0;i<4;i++)for(int j=0;j<4;j++)terms[group][j]+=w[i]*pg_param(PG_P_LIGHT+16*group+4*i+j);
    }
    light=terms[0];chroma=terms[1];
}
float pg_carry_energy(float energy,float oldc,float newc,float dh) {
    float b0=sqrt(66.0*oldc),b=sqrt(66.0*newc);
    float u0=pg_div(energy,length(vec2(sqrt(abs(energy)),b0))+b0);
    float db=pg_div(66.0*(newc-oldc),b+b0),du=dh-db,u=u0+du,r=abs(u)+abs(u0);
    float slope=u*u0>=0.0?r:pg_div(u*u+u0*u0,r);
    float change=du*(slope+2.0*b)+2.0*u0*db;
    return abs(change)<=.25*abs(energy)?energy+change:u*(abs(u)+2.0*b);
}
float pg_web_a(float c,float toe) {
    float m=max(c,toe),x=c/m,y=toe/m;
    return y/sqrt(length(vec2(x*x,y*y)));
}
float pg_web_complement(float c,float toe) {
    if(c>toe)return 1.0-pg_web_a(c,toe);
    float x=c/toe,z=x*x*x*x,w=sqrt(sqrt(1.0+z));
    return z/(w*(w+1.0)*(w*w+1.0));
}
float pg_web_radial(float c,float g,float toe) {
    return c*(g>=1.0?1.0+(g-1.0)*pg_web_complement(c,toe):g+(1.0-g)*pg_web_a(c,toe));
}
float pg_web_coefficient(float source,float target,float toe) {
    float a=pg_web_a(source,toe),r=target/source;
    return r>=1.0?1.0+(r-1.0)/(1.0-a):(r-a)/(1.0-a);
}
float pg_eighth(float x) {float q=x*x;q*=q;return q*q;}
float pg_web_toe(float a[6],float b[6],bool paired) {
    float m=2.0;
    for(int i=0;i<6;i++) {m=min(m,.1*a[i]);if(paired)m=min(m,.1*b[i]);}
    float total=pg_eighth(m/2.0);
    for(int i=0;i<6;i++) {total+=pg_eighth(m/(.1*a[i]));if(paired)total+=pg_eighth(m/(.1*b[i]));}
    return m/sqrt(sqrt(sqrt(total)));
}
void pg_web_sector(float angle,float knots[7],out int index,out float t,out float local) {
    local=mod(angle-knots[0],6.283185307179586)+knots[0];index=0;
    for(int i=1;i<6;i++)if(local>=knots[i])index=i;
    t=(local-knots[index])/(knots[index+1]-knots[index]);
}
float pg_web_field(float angle,float knots[7],float values[6]) {
    int i;float t,local;pg_web_sector(angle,knots,i,t,local);
    float a=values[i],b=values[(i+1)%6];return a+(b-a)*pg_smoother(t);
}
float pg_web_angle(float angle,float src[7],float dst[7]) {
    bool same=true;for(int i=0;i<6;i++)same=same&&(src[i]==dst[i]);if(same)return angle;
    int index;float t,local;pg_web_sector(angle,src,index,t,local);
    float slopes[6],d[7];
    for(int i=0;i<6;i++)slopes[i]=(dst[i+1]-dst[i])/(src[i+1]-src[i]);
    for(int i=0;i<6;i++) {
        float a=slopes[i],b=slopes[(i+5)%6],m=min(a,b),x=m/a,y=m/b;
        d[i]=m/sqrt(sqrt(.5*(x*x*x*x+y*y*y*y)));
    }
    d[6]=d[0];float width=src[index+1]-src[index],a=dst[index],b=dst[index+1];
    float da=width*d[index]/5.0,db=width*d[index+1]/5.0,v[6];
    v[0]=a;v[1]=a+da;v[2]=a+2.0*da;v[3]=b-2.0*db;v[4]=b-db;v[5]=b;
    for(int level=5;level>0;level--)for(int i=0;i<level;i++)v[i]=(1.0-t)*v[i]+t*v[i+1];
    return angle+(v[0]-local);
}
void pg_web(inout vec3 result,inout float energy,float magnitude,out float final_scale) {
    final_scale=1.0;if(pg_param(PG_P_WEB_ACTIVE)==0.0)return;
    int row;float w;pg_cardinal(magnitude,row,w);
    float src[7],dst[7],r[6],target[6],finalr[6],gc[6],gs[6];
    for(int i=0;i<6;i++) {
        src[i]=pg_blend_control(PG_P_ANGLES,row,i,6,w);
        dst[i]=src[i]+pg_blend_control(PG_P_HUE_RADIANS,row,i,6,w);
        r[i]=exp(pg_blend_control(PG_P_LOG_RADII,row,i,6,w));
        target[i]=r[i]*exp2(pg_blend_control(PG_P_LOG_CHROMA,row,i,6,w));
        finalr[i]=target[i]*exp2(pg_blend_control(PG_P_LOG_SCALE,row,i,6,w));
    }
    src[6]=src[0]+6.283185307179586;dst[6]=dst[0]+6.283185307179586;
    float at=pg_web_toe(r,r,false),ct=pg_web_toe(r,target,true),st=pg_web_toe(target,finalr,true);
    for(int i=0;i<6;i++) {gc[i]=log(pg_web_coefficient(r[i],target[i],ct));gs[i]=log(pg_web_coefficient(target[i],finalr[i],st));}
    float c=length(result.yz),angle=pg_atan2(result.z,result.y);
    float outangle=angle+pg_smoother(c/at)*(pg_web_angle(angle,src,dst)-angle);
    float c1=pg_web_radial(c,exp(pg_web_field(outangle,dst,gc)),ct);
    energy=pg_carry_energy(energy,c,c1,0.0);
    result.yz=outangle==angle?result.yz*pg_div(c1,c):c1*vec2(cos(outangle),sin(outangle));
    float c2=pg_web_radial(c1,exp(pg_web_field(outangle,dst,gs)),st);
    if(c1>0.0)final_scale=c2/c1;
}
vec4 pg_grade(vec4 q) {
    if(pg_param(PG_P_LIGHT_ACTIVE)==0.0&&pg_param(PG_P_CHROMA_ACTIVE)==0.0&&pg_param(PG_P_ADAPTANCE_ACTIVE)==0.0
       &&pg_param(PG_P_WEB_ACTIVE)==0.0&&pg_param(PG_P_GAIN)==0.0)return q;
    float c=length(q.yz),magnitude=sqrt(abs(q.w)+66.0*c);if(magnitude==0.0)return q;
    float r=PG_CHROMA_CROSSOVER*pg_physical_capacity(q.w,q.yz),normal=max(c,r);
    float cn=pg_div(c,normal),rn=pg_div(r,normal),den=cn*cn+rn*rn;
    float visibility=c/length(vec2(c,2.0));visibility*=visibility;
    vec4 drivers=vec4(visibility*pg_div(rn*rn,den),visibility*pg_div(cn*cn,den),q.yz/length(vec2(c,2.0)));
    vec4 light,chroma;pg_terms_height(magnitude,light,chroma);
    vec3 result=vec3(q.x+dot(light,drivers),q.yz*exp2(pg_param(PG_P_GAIN)+dot(chroma,drivers)));
    float energy=pg_carry_energy(q.w,c,length(result.yz),result.x-q.x),final_scale;
    pg_web(result,energy,magnitude,final_scale);
    if(pg_param(PG_P_ADAPTANCE_ACTIVE)!=0.0)pg_adaptance(result,energy,magnitude,c,q.w);
    result.yz*=final_scale;energy*=final_scale*final_scale;
    if(final_scale!=1.0)result.x=pg_hk(sign(energy)*sqrt(abs(energy)),length(result.yz));
    return vec4(result,energy);
}
bool pg_decode(vec4 q,out vec3 rgb) {
    float ja=sign(q.w)*sqrt(abs(q.w)),t=sign(ja)*pg_power(abs(ja)/PG_WHITE,1.0/PG_GAMMA);
    float c=length(q.yz);vec2 co=c>0.0?q.yz/c:vec2(0.0);
    vec3 axis=PG_AW/(1204.0*pg_eccentricity(co.x,co.y))*(co.x*vec3(451.0,-891.0,-220.0)/1403.0+co.y*vec3(288.0,-261.0,-6300.0)/1403.0);
    vec3 response=PG_RW*t+c*axis;
    rgb=vec3(0.0);
    if(any(isnan(response))||any(isinf(response))||any(greaterThanEqual(abs(response),vec3(400.0))))return false;
    if(t>.5&&all(greaterThanEqual(response,vec3(0.0)))) {
        vec3 deficit=(1.0-t)-c*axis/PG_RW,cones;
        for(int i=0;i<3;i++)cones[i]=-1024.0*pg_expm1((50.0/21.0)*(pg_log1p(-deficit[i])-pg_log1p(PG_RW/(400.0-PG_RW)*deficit[i])));
        rgb=1024.0-pg_native(cones,PG_TO_RGB_0,PG_TO_RGB_2);
    } else {
        vec3 cones=sign(response)*1024.0*pg_power(abs(response)*(400.0-PG_RW)/(PG_RW*(400.0-abs(response))),vec3(50.0/21.0));
        rgb=pg_native(cones,PG_TO_RGB_0,PG_TO_RGB_2);
    }
    return !(any(isnan(rgb))||any(isinf(rgb)));
}
bool pg_cube_candidate(float energy,vec2 direction,float radius,out vec3 rgb) {
    vec4 q=vec4(sqrt(energy+66.0*radius),radius*direction,energy);
    return pg_decode(q,rgb)&&all(greaterThanEqual(rgb,vec3(0.0)))&&all(lessThanEqual(rgb,vec3(1024.0)));
}
vec4 pg_export_state(vec4 q,out vec3 rgb) {
    if(q.w<=0.0){rgb=vec3(0.0);return vec4(0.0);}
    if(q.w>=PG_WHITE*PG_WHITE){rgb=vec3(1024.0);return vec4(PG_WHITE,0.0,0.0,PG_WHITE*PG_WHITE);}
    if(pg_decode(q,rgb)&&all(greaterThanEqual(rgb,vec3(0.0)))&&all(lessThanEqual(rgb,vec3(1024.0))))return q;
    float c=length(q.yz),ja=sqrt(q.w),t=pg_power(ja/PG_WHITE,1.0/PG_GAMMA);
    vec2 co=c>0.0?q.yz/c:vec2(0.0);
    vec3 axis=PG_AW/(1204.0*pg_eccentricity(co.x,co.y))*(co.x*vec3(451.0,-891.0,-220.0)/1403.0+co.y*vec3(288.0,-261.0,-6300.0)/1403.0);
    float high=c;
    for(int i=0;i<3;i++) {
        if(axis[i]>0.0)high=min(high,(PG_RESPONSE_HIGH[i]-PG_RW*t)/axis[i]);
        if(axis[i]<0.0)high=min(high,(PG_RESPONSE_LOW[i]-PG_RW*t)/axis[i]);
    }
    float low=0.0;vec3 saved;
    pg_cube_candidate(q.w,co,0.0,saved);
    float seed=pg_physical_capacity(q.w,q.yz);
    if(seed>0.0&&seed<high) {
        if(pg_cube_candidate(q.w,co,seed,rgb)){low=seed;saved=rgb;}else high=seed;
    }
    if(pg_cube_candidate(q.w,co,high,rgb))return vec4(sqrt(q.w+66.0*high),high*co,q.w);
    for(int step=0;step<24;step++) {
        float middle=low+(high-low)*.5;if(middle==low||middle==high)break;
        if(pg_cube_candidate(q.w,co,middle,rgb)){low=middle;saved=rgb;}else high=middle;
    }
    rgb=saved;return vec4(sqrt(q.w+66.0*low),low*co,q.w);
}
vec3 pg_export(vec4 q) {
    vec3 rgb;pg_export_state(q,rgb);return rgb;
}
vec4 pg_hk_guard(vec4 q,float reference_energy) {
    if(pg_param(PG_P_HK_GUARD)==0.0||q.x<0.0||any(isnan(q))||any(isinf(q)))return q;
    float c=length(q.yz),b=sqrt(66.0*c),d=q.w;
    // Signed energy differs from H^2-66C below the H-K boundary. Recover the
    // latter without subtracting rounded, nearly equal H^2 and 66C values.
    if(d<0.0) {
        float u=pg_div(d,length(vec2(sqrt(abs(d)),b))+b);
        d+=2.0*u*u;
    }
    float x=q.x*q.x,e0=max(reference_energy,0.0);
    float low=min(e0,x),high=max(e0,x),delta=pg_div(low,1.0+pg_div(low,high));
    if(d>=delta)return q;
    float v=pg_div(delta,delta+max(delta-d,0.0)),u=1.0-v;
    float e=delta*sqrt(v)/sqrt(sqrt(v*v+4.0*v*u+10.0*u*u));
    float z=max(e-d,0.0),l=.44*x,m=max(2.0*l,z);
    float lift=l*pg_div(pg_div(z,m),length(vec2(pg_div(2.0*l,m),pg_div(z,m))));
    float cg=min(c,max(0.0,(x+lift-e)/66.0));
    return vec4(sqrt(max(x+lift,0.0)),q.yz*(c>0.0?cg/c:1.0),e);
}
vec3 pg_export_reference(vec4 q,vec4 reference) {
    return pg_export(pg_hk_guard(q,reference.w));
}
vec3 pg_forward(vec3 rgb) {
    if(pg_param(PG_P_IDENTITY)!=0.0)return rgb;
    vec4 toned=pg_tone(pg_enter(rgb));
    return pg_export_reference(pg_grade(toned),toned);
}
void main() {
    int id=int(gl_GlobalInvocationID.x);if(id>=sample_count)return;
    ivec2 address=ivec2(id%256,id/256);vec4 input_value=texelFetch(input_pixels,address,0),result;
    if(operation==0)result=pg_enter(input_value.xyz);
    else if(operation==2&&pg_param(PG_P_IDENTITY)!=0.0)result=vec4(input_value.xyz,1.0);
    else {
        vec4 state=operation==1?input_value:pg_enter(input_value.xyz);
        vec4 toned=pg_tone(state);
        result=vec4(pg_export_reference(pg_grade(toned),toned),1.0);
    }
    imageStore(output_pixels,address,result);
}
'''


class Program(boundary_gpu.Program):
    def __init__(self,kernel):
        super().__init__(kernel.data,source,True,kernel.table.expanded)
        self.set_kernel(kernel)

    def set_kernel(self,kernel):
        values=parameters(kernel)
        self.parameters=self.gpu.types.GPUTexture((256,1),format='R32F',
            data=self.gpu.types.Buffer('FLOAT',(1,256),values))
