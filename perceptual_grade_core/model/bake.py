"""Portable float32 regular-Grade kernel, checked against the float64 oracle.

The boundary/table policy is shared with GLSL. This module has no Blender or
Colour dependency; control constants are supplied by the independent model.
"""
import hashlib
import numpy as np
from .boundary_polynomial import PolynomialTable,LOOKUP_KEY

GAMMA=np.float32(.525*(1.48+np.sqrt(.1)))
CHROMA_CROSSOVER=np.float32(.4)
INVERSE=(np.array([[451.,288.],[-891.,-261.],[-220.,-6300.]])/1403).astype('f4')


def divide(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.zeros_like(a),where=b!=0)


def eccentricity(c,s):
    c2=c*c-s*s;s2=2*c*s;c3=c2*c-s2*s;s3=s2*c+c2*s
    c4=c2*c2-s2*s2;s4=2*c2*s2
    return 1-.0582*c-.0258*c2-.1347*c3+.0289*c4-.1475*s-.0308*s2+.0385*s3+.0096*s4


def expm1(x):
    """Shared float32 cancellation-free form, including GLSL without expm1."""
    x=np.asarray(x,dtype=np.float32)
    z=np.where(abs(x)<.02,x,0.)
    small=z*(1+z*(.5+z*(1/6+z*(1/24+z/120))))
    return np.where(abs(x)<.02,small,np.exp(x)-1)


def log1p(x):
    x=np.asarray(x,dtype=np.float32)
    z=np.where(abs(x)<.05,x,0.)
    small=z*(1+z*(-.5+z*(1/3+z*(-.25+z*(.2+z*(-1/6+z/7))))))
    with np.errstate(divide='ignore'):
        return np.where(abs(x)<.05,small,np.log(1+x))


def softplus(x):
    return np.maximum(x,0)+log1p(np.exp(-abs(x)))


def integral(t,left,right,width,constant=None):
    if constant is None:
        gap=(right-left)/width
        constant=gap+np.log(-expm1(-gap))-log1p(np.exp(-left/width))
    with np.errstate(divide='ignore'):
        logq=constant-softplus((right-t)/width)+np.log(-expm1(-t/width))
    return width*softplus(logq)


def kernels(t,knots,width):
    t=np.asarray(t,dtype=np.float32)
    return np.stack([integral(t,l,r,width)/integral(np.float32(1),l,r,width)
                     for l,r in zip(knots[:-1],knots[1:])],-1)


class PreparedKernels:
    """Grade constants for the existing float32 integral, without tabulation."""
    def __init__(self,knots,width):
        self.knots=np.asarray(knots,dtype=np.float32);self.width=np.float32(width)
        self.constants=np.array([(r-l)/self.width+np.log(-expm1(-(r-l)/self.width))-
            log1p(np.exp(-l/self.width)) for l,r in zip(self.knots[:-1],self.knots[1:])],dtype=np.float32)
        self.normalizers=np.array([integral(np.float32(1),l,r,self.width,v)
            for l,r,v in zip(self.knots[:-1],self.knots[1:],self.constants)],dtype=np.float32)
        self.endpoints=self.evaluate(np.array([0.,1.],dtype=np.float32))

    def evaluate(self,t):
        t=np.asarray(t,dtype=np.float32)
        with np.errstate(divide='ignore'):
            tail=np.log(-expm1(-t/self.width))
        return np.stack([self.width*softplus(v-softplus((r-t)/self.width)+tail)/n
            for r,v,n in zip(self.knots[1:],self.constants,self.normalizers)],-1)

    def evaluate_bounded(self,t):
        """Reuse exact prepared endpoint values for clipped tonal coordinates."""
        t=np.asarray(t,dtype=np.float32);upper=t==1;edge=(t==0)|upper
        if not np.any(edge):return self.evaluate(t)
        result=self.endpoints[upper.astype(np.intp)]
        interior=~edge
        if np.any(interior):result[interior]=self.evaluate(t[interior])
        return result


def weights(u,adapt=False):
    u=np.minimum.accumulate(np.clip(u,0,1),axis=-1)
    first=u[...,0] if adapt else np.ones(u.shape[:-1],dtype=np.float32)
    return np.stack((first-u[...,1],u[...,1]-u[...,2],u[...,2]-u[...,3],u[...,3]),-1)


def smoother(t):
    t=np.clip(np.asarray(t,dtype='f4'),0,1)
    return t*t*t*(10+t*(-15+6*t))


def carry_energy(energy,old_c,new_c,dh=0):
    b0=np.sqrt(66*old_c);b=np.sqrt(66*new_c)
    u0=divide(energy,np.hypot(np.sqrt(abs(energy)),b0)+b0)
    db=divide(66*(new_c-old_c),b+b0);du=dh-db;u=u0+du;r=abs(u)+abs(u0)
    slope=np.where(u*u0>=0,r,divide(u*u+u0*u0,r))
    change=du*(slope+2*b)+2*u0*db
    return np.where(abs(change)<=.25*abs(energy),energy+change,u*(abs(u)+2*b))


def response_transform(response,gain):
    """Signed cone-gain action, including the algebraic exterior of ±400."""
    r,m=np.broadcast_arrays(np.asarray(response,dtype='f4'),np.asarray(gain,dtype='f4'))
    t=np.minimum(abs(r)/400,1)
    result=r*(m/((1-t)+m*t))
    outside=abs(r)>400
    if np.any(outside):
        y=abs(r[outside])/400-1;root=np.sqrt(y)
        u=root*(2*root/(np.hypot(1,2*root)+1));z=u/m[outside]
        result=np.array(result,copy=True)
        result[outside]=np.sign(r[outside])*400*(1+z+z*z)
    return np.where(m==1,r,result)


def response_delta(response,log_gain):
    """Evaluate T_exp(log_gain)(r)-r without cancelling near the identity."""
    r,l=np.broadcast_arrays(np.asarray(response,dtype='f4'),np.asarray(log_gain,dtype='f4'))
    m=np.exp(l);t=np.minimum(abs(r)/400,1)
    result=r*expm1(l)*(1-t)/((1-t)+m*t)
    outside=abs(r)>400
    if np.any(outside):
        root=np.sqrt(abs(r[outside])/400-1)
        u=root*(2*root/(np.hypot(1,2*root)+1));z=u/m[outside]
        result=np.array(result,copy=True)
        result[outside]=np.sign(r[outside])*400*u*expm1(-l[outside])*(1+u+z)
    return result


def reach_weight(chroma,energy,half_saturation):
    """Frozen source-saturation mask, half active at direct C/|J_A| radius."""
    c,e=np.broadcast_arrays(np.asarray(chroma,dtype='f4'),np.asarray(energy,dtype='f4'))
    a=np.float32(half_saturation)*np.sqrt(abs(e))
    scale=np.maximum(a,c);an=divide(a,scale);cn=divide(c,scale)
    an2=an*an;cn2=cn*cn
    return np.where(scale>0,divide(an2*an2,an2*an2+cn2*cn2),np.float32(1))


def web_toe(*radii):
    values=np.concatenate((np.full(radii[0].shape[:-1]+(1,),2,dtype='f4'),
                           *[np.float32(.1)*r for r in radii]),axis=-1)
    small=np.min(values,axis=-1)
    return small/np.power(np.sum((small[...,None]/values)**8,axis=-1),np.float32(.125))


def web_attenuation(c,toe):
    scale=np.maximum(c,toe)
    return (toe/scale)/np.sqrt(np.hypot((c/scale)**2,(toe/scale)**2))


def web_complement(c,toe):
    fourth=(np.minimum(c,toe)/toe)**4;w=np.sqrt(np.sqrt(1+fourth))
    return np.where(c<=toe,fourth/(w*(w+1)*(w*w+1)),1-web_attenuation(c,toe))


def web_coefficient(source,target,toe):
    a=web_attenuation(source,toe);ratio=target/source
    return np.where(ratio>=1,1+(ratio-1)/(1-a),(ratio-a)/(1-a))


def web_radial(c,g,toe):
    gain=np.where(g>=1,1+(g-1)*web_complement(c,toe),g+(1-g)*web_attenuation(c,toe))
    return c*gain


def web_sector(angle,knots):
    period=np.float32(2*np.pi)
    local=np.remainder(angle-knots[...,0],period)+knots[...,0]
    extended=np.concatenate((knots,knots[...,:1]+period),axis=-1)
    index=np.minimum(np.sum(local[...,None]>=extended[...,1:],axis=-1),5)
    left=web_gather(extended,index);width=web_gather(extended,index+1)-left
    return index,(local-left)/width,width,local


def web_gather(values,index):
    return np.take_along_axis(values,index[...,None],axis=-1)[...,0]


def web_field(angle,knots,values):
    index,t,_,_=web_sector(angle,knots)
    ext=np.concatenate((values,values[...,:1]),axis=-1)
    a=web_gather(ext,index);b=web_gather(ext,index+1)
    return a+(b-a)*smoother(t)


def web_angle(angle,source,target):
    index,t,width,local=web_sector(angle,source)
    period=np.float32(2*np.pi)
    src=np.concatenate((source,source[...,:1]+period),axis=-1)
    dst=np.concatenate((target,target[...,:1]+period),axis=-1)
    slopes=np.diff(dst,axis=-1)/np.diff(src,axis=-1);prev=np.roll(slopes,1,axis=-1)
    small=np.minimum(slopes,prev)
    d=small/np.sqrt(np.sqrt(np.float32(.5)*((small/slopes)**4+(small/prev)**4)))
    d=np.concatenate((d,d[...,:1]),axis=-1)
    a=web_gather(dst,index);b=web_gather(dst,index+1)
    da=width*web_gather(d,index)/5;db=width*web_gather(d,index+1)/5
    values=[a,a+da,a+2*da,b-2*db,b-db,b]
    for _ in range(5):values=[(1-t)*x+t*y for x,y in zip(values[:-1],values[1:])]
    return np.where(np.all(source==target,axis=-1),angle,angle+values[0]-local)


class Kernel:
    def __init__(self,data,tone,color,table=None,*,hk_guard=False):
        self.data=data
        self.hk_guard=bool(hk_guard)
        self.table=PolynomialTable(data) if table is None else table
        self.tone=tone;self.color=color.settings
        p=self.color
        self.controls={name:np.asarray(getattr(p,name),dtype=np.float32)
            for name in ('lightness','chroma','adaptance')}
        self.active_controls={name:bool(np.any(values)) for name,values in self.controls.items()}
        self.directional_drivers=any(np.any(self.controls[name][:,2:]) for name in ('lightness','chroma'))
        self.tone_curve=PreparedKernels(tone.knots,float(tone.width))
        self.tone_curve_residual=np.asarray(tone.residual,dtype=np.float32)
        fits=(p.lightness_fit,p.chroma_fit)
        self.coefficient_curves={fit:PreparedKernels(tone.coefficient_knots,.05*2**(-fit)) for fit in dict.fromkeys(fits)}
        self.control_fits=dict(zip(('lightness','chroma'),fits))
        self.web_active=bool(p.web.active)
        self.anchors=np.asarray(tone.target[1:]*100,dtype='f4')
        self.adaptance_log_gains=np.asarray(color.adaptation.log_gains,dtype='f4')
        self.adaptance_half_saturation=np.float32(color.adaptation.half_saturation)
        self.web_controls={name:np.asarray(getattr(p.web,name),dtype='f4') for name in
                           ('log_chroma','hue_degrees','log_scale')}
        self.web_radii=np.zeros((4,6),dtype='f4');self.web_angles=self.web_radii.copy()
        if self.web_active:
            geometry=color.web.geometry
            self.web_radii=np.asarray(np.log(geometry.radii),dtype='f4')
            self.web_angles=np.asarray(geometry.angles,dtype='f4')
            prepared=self.web_parameters(self.anchors)
            for angles in prepared[:2]:
                gaps=np.diff(np.concatenate((angles,angles[...,:1]+np.float32(2*np.pi)),axis=-1),axis=-1)
                if np.any(gaps<=0):raise ValueError('Web hue intervals must remain ordered in float32')
            if any(not np.isfinite(v).all() for v in prepared) or any(np.any(v<=0) for v in prepared[2:]):
                raise ValueError('Web geometry exceeds the finite positive float32 range')
        self.resource_key=hashlib.sha256(LOOKUP_KEY+b'warp-v2'+b''.join(np.asarray(data[key]).tobytes() for key in sorted(data))).hexdigest()
        self.color_active=any(self.active_controls.values()) or p.global_gain!=0 or self.web_active
        self.identity=not tone.active and not self.color_active
        self.white=np.float32(data['white']);self.fl=np.float32(data['fl']);self.aw=np.float32(data['aw'])
        self.matrix=np.asarray(data['matrices'][0],dtype=np.float32)
        self.inverse=np.linalg.inv(np.asarray(data['matrices'][0],dtype=np.float64)).astype('f4')
        p=np.power(self.fl*1024,np.float32(.42))
        self.rw=np.float32(400*p/(27.13+p))
        effective=np.column_stack((self.matrix[:,0],1-self.matrix[:,0]-self.matrix[:,2],self.matrix[:,2])).astype(float)
        inverse=np.linalg.inv(effective)
        limits=np.stack((1024*np.minimum(inverse,0).sum(axis=1),1024*np.maximum(inverse,0).sum(axis=1)))
        pp=np.power(abs(limits)*float(self.fl),.42)
        self.response_limits=(np.sign(limits)*400*pp/(27.13+pp)).astype('f4')

    @staticmethod
    def native(v,matrix):
        return v[...,1,None]+(v[...,0]-v[...,1])[...,None]*matrix[:,0]+(v[...,2]-v[...,1])[...,None]*matrix[:,2]

    def enter(self,rgb):
        """Encode native RGB directly into raw signed modCAM16 coordinates."""
        rgb=np.asarray(rgb,dtype=np.float32)
        cones=self.native(rgb,self.inverse)
        p=np.power(abs(cones)*self.fl,np.float32(.42));r=np.sign(cones)*400*p/(27.13+p)
        t=(2*r[...,0]+r[...,1]+.05*r[...,2])/(3.05*self.rw)
        a=(r[...,0]-r[...,1])+(r[...,2]-r[...,1])/11
        b=((r[...,0]-r[...,2])+(r[...,1]-r[...,2]))/9
        radius=np.hypot(a,b);co=divide(a,radius);si=divide(b,radius)
        c=1204*eccentricity(co,si)*radius/self.aw
        ja=np.sign(t)*self.white*np.power(abs(t),GAMMA)
        base=np.sqrt(66*c);magnitude=np.hypot(ja,base)
        h=np.where(ja>=0,magnitude,2*base-magnitude)
        return np.stack((h,c*co,c*si,ja*abs(ja)),-1).astype('f4')

    def capacity(self,energy,ab):
        """Physical fitted AP1 capacity, zero at tips and outside its domain."""
        energy=np.asarray(energy,dtype=np.float32)
        positive=(energy>0)&(energy<self.white*self.white)
        t=np.power(np.sqrt(np.maximum(energy,0))/self.white,1/GAMMA)
        t=np.clip(t,0,1);angle=np.arctan2(ab[...,1],ab[...,0])
        return np.where(positive,self.rw*t*(1-t)*self.table.capacity(t,angle),0.)

    def tone_state(self,state):
        if not self.tone.active:return state.copy()
        tone=self.tone;q=np.asarray(state,dtype=np.float32);ja=np.sign(q[...,3])*np.sqrt(abs(q[...,3]));c=np.hypot(q[...,1],q[...,2])
        magnitude=np.sqrt(ja*ja+66*c)
        t=np.maximum(0,(magnitude/100-np.float32(tone.black))/np.float32(tone.span))
        k=self.tone_curve.evaluate(t)
        # A matvec can select a different float32 reduction for different batch
        # sizes. Candidate validation and returned-RGB validation must evaluate
        # the same per-pixel curve even when accepted rows have been removed.
        h=100*(np.float32(tone.slope)*t+np.sum(k*self.tone_curve_residual,axis=-1))
        h=np.where(t==1,np.float32(tone.high),h)
        denominator=np.hypot(66*c,2*ja*h)+66*c
        gain=divide(2*h*h,denominator);j=ja*gain;ab=q[...,1:3]*gain[...,None]
        base=np.sqrt(66*np.hypot(ab[...,0],ab[...,1]));magnitude=np.hypot(j,base)
        h=np.where(j>=0,magnitude,2*base-magnitude)
        return np.concatenate((h[...,None],ab,(j*abs(j))[...,None]),-1)

    def tonal(self,state,fit,*,height=None):
        h=np.sqrt(abs(state[...,3])+66*np.hypot(state[...,1],state[...,2])) if height is None else height
        t=np.clip(h/np.float32(self.tone.high),0,1);t=t*t*t*(10+t*(-15+6*t))
        return self.coefficient_curves[fit].evaluate_bounded(t)

    def terms(self,state=None,tonal=None,*,height=None):
        q=None if state is None else np.asarray(state,dtype=np.float32)
        cache={} if tonal is None else tonal
        weighted={}
        def term(name):
            if not self.active_controls[name]:return None
            fit=self.control_fits[name]
            if fit not in cache:cache[fit]=self.tonal(q,fit,height=height)
            if fit not in weighted:weighted[fit]=weights(cache[fit])
            return np.sum(weighted[fit][...,None]*self.controls[name],axis=-2)
        return term('lightness'),term('chroma')

    def cardinal(self,height,values):
        index=np.clip(np.searchsorted(self.anchors,height,side='right')-1,0,2)
        t=smoother((height-self.anchors[index])/(self.anchors[index+1]-self.anchors[index]))
        return values[index]+(values[index+1]-values[index])*t[...,None]

    def web_parameters(self,height):
        angles=self.cardinal(height,self.web_angles)
        radii=np.exp(self.cardinal(height,self.web_radii))
        target_angles=angles+self.cardinal(height,self.web_controls['hue_degrees'])*np.float32(np.pi/180)
        target=radii*np.exp2(self.cardinal(height,self.web_controls['log_chroma']))
        final=target*np.exp2(self.cardinal(height,self.web_controls['log_scale']))
        angular_toe=web_toe(radii);chroma_toe=web_toe(radii,target);scale_toe=web_toe(target,final)
        gc=web_coefficient(radii,target,chroma_toe[...,None])
        gs=web_coefficient(target,final,scale_toe[...,None])
        return angles,target_angles,angular_toe,chroma_toe,scale_toe,gc,gs

    def to_responses(self,q,energy):
        """Recover raw signed responses without rejecting the response poles."""
        c=np.hypot(q[...,1],q[...,2]);co=divide(q[...,1],c);si=divide(q[...,2],c)
        axis=(self.aw/(1204*eccentricity(co,si)))[...,None]*(
            co[...,None]*INVERSE[:,0]+si[...,None]*INVERSE[:,1])
        ja=np.sign(energy)*np.sqrt(abs(energy))
        neutral=self.aw/np.float32(3.05)*np.sign(ja)*np.power(abs(ja)/100,1/GAMMA)
        return neutral[...,None]+c[...,None]*axis

    def from_responses(self,r):
        achromatic=(2*r[...,0]+r[...,1])+.05*r[...,2]
        a=(r[...,0]-r[...,1])+(r[...,2]-r[...,1])/11
        b=((r[...,0]-r[...,2])+(r[...,1]-r[...,2]))/9
        return self.from_components(achromatic,a,b)

    def from_components(self,achromatic,a,b):
        radius=np.hypot(a,b);co=divide(a,radius);si=divide(b,radius)
        c=1204*eccentricity(co,si)*radius/self.aw
        ja=100*np.sign(achromatic)*np.power(abs(achromatic)/self.aw,GAMMA)
        base=np.sqrt(66*c);magnitude=np.hypot(ja,base)
        h=np.where(ja>=0,magnitude,2*base-magnitude)
        return np.stack((h,c*co,c*si),-1),ja*abs(ja)

    def adapt_state(self,q,energy,height,source_chroma,source_energy):
        logs=self.cardinal(height,self.adaptance_log_gains);m=np.exp(logs)
        r0=self.aw/np.float32(3.05)*np.power(height/100,1/GAMMA)
        t=np.minimum(r0/400,1)[...,None]
        ratios=m/((1-t)+m*t)
        outside=r0>400
        if np.any(outside):
            ratios=np.array(ratios,copy=True)
            ratios[outside]=response_transform(r0[outside,None],m[outside])/r0[outside,None]
        k=np.float32(3.05)/((2*ratios[...,0]+ratios[...,1])+.05*ratios[...,2])
        v=reach_weight(source_chroma,source_energy,self.adaptance_half_saturation)
        gains=np.exp(v[...,None]*logs);factor=np.exp(v*np.log(k))
        # Transport only the response deltas. Re-forming two nearly equal
        # neutral responses would discard the small tint in float32.
        delta=response_delta(self.to_responses(q,energy),v[...,None]*logs)
        c=np.hypot(q[...,1],q[...,2]);co=divide(q[...,1],c);si=divide(q[...,2],c)
        raw_scale=self.aw/(1204*eccentricity(co,si))
        a=(q[...,1]*raw_scale+(delta[...,0]-delta[...,1])+(delta[...,2]-delta[...,1])/11)*factor
        b=(q[...,2]*raw_scale+((delta[...,0]-delta[...,2])+(delta[...,1]-delta[...,2]))/9)*factor
        achromatic=self.aw*np.sign(energy)*np.power(np.sqrt(abs(energy))/100,1/GAMMA)
        achromatic=(achromatic+(2*delta[...,0]+delta[...,1])+.05*delta[...,2])*factor
        out,e=self.from_components(achromatic,a,b)
        neutral=(source_chroma==0)&np.all(q[...,1:]==0,axis=-1)
        e=np.where(neutral,energy,e)
        c=np.hypot(out[...,1],out[...,2]);base=np.sqrt(66*c);mag=np.hypot(np.sqrt(abs(e)),base)
        out[...,0]=np.where(e>=0,mag,2*base-mag)
        unchanged=(v==0)|(np.all(gains==1,axis=-1)&(factor==1))|((energy==0)&np.all(q==0,axis=-1))
        return np.where(unchanged[...,None],q,out),np.where(unchanged,energy,e)

    def grade(self,state,tonal=None):
        q=np.asarray(state,dtype=np.float32)
        if not self.color_active:return q.copy()
        p=self.color;c=np.hypot(q[...,1],q[...,2]);magnitude=np.sqrt(abs(q[...,3])+66*c)
        r=CHROMA_CROSSOVER*self.capacity(q[...,3],q[...,1:3]);scale=np.maximum(c,r)
        cn=divide(c,scale);rn=divide(r,scale);den=cn*cn+rn*rn;visibility=(c/np.hypot(c,2))**2
        drivers=np.concatenate((np.stack((visibility*divide(rn*rn,den),visibility*divide(cn*cn,den)),-1),q[...,1:3]/np.hypot(c,2)[...,None]),-1)
        light,chroma=self.terms(q,tonal,height=magnitude)
        loggain=np.full_like(c,p.global_gain)
        if chroma is not None:loggain+=np.sum(chroma*drivers,axis=-1)
        result=q[...,:3].copy();result[...,1:]*=np.exp2(loggain)[...,None]
        if light is not None:result[...,0]+=np.sum(light*drivers,axis=-1)
        c1=np.hypot(result[...,1],result[...,2])
        energy=carry_energy(q[...,3],c,c1,result[...,0]-q[...,0])
        radial_scale=np.ones_like(c1)
        if self.web_active:
            src,dst,at,ct,st,gc,gs=self.web_parameters(magnitude)
            angle=np.arctan2(result[...,2],result[...,1])
            warped=web_angle(angle,src,dst)
            out_angle=angle+smoother(c1/at)*(warped-angle)
            chroma_gain=np.exp(web_field(out_angle,dst,np.log(gc)))
            c2=web_radial(c1,chroma_gain,ct)
            energy=carry_energy(energy,c1,c2)
            ab=np.stack((c2*np.cos(out_angle),c2*np.sin(out_angle)),-1)
            unchanged_angle=out_angle==angle
            result[...,1:]=np.where(unchanged_angle[...,None],result[...,1:]*divide(c2,c1)[...,None],ab)
            scale_gain=np.exp(web_field(out_angle,dst,np.log(gs)))
            c3=web_radial(c2,scale_gain,st)
            radial_scale=np.where(c2>0,divide(c3,c2),np.float32(1))
            c1=c2
        if self.active_controls['adaptance']:
            result,energy=self.adapt_state(result,energy,magnitude,c,q[...,3])
        result[...,1:]*=radial_scale[...,None];energy*=radial_scale*radial_scale
        base=np.sqrt(66*np.hypot(result[...,1],result[...,2]));mag=np.hypot(np.sqrt(abs(energy)),base)
        result[...,0]=np.where(radial_scale==1,result[...,0],np.where(energy>=0,mag,2*base-mag))
        output=np.concatenate((result,energy[...,None]),-1).astype('f4')
        return np.where((magnitude==0)[...,None],q,output)

    def decode(self,state):
        """Actual raw decoder; unsafe response-pole trials return nonfinite RGB."""
        q=np.asarray(state,dtype=np.float32);e=q[...,3]
        ja=np.sign(e)*np.sqrt(abs(e));t=np.sign(ja)*np.power(abs(ja)/self.white,1/GAMMA)
        c=np.hypot(q[...,1],q[...,2]);co=divide(q[...,1],c);si=divide(q[...,2],c)
        axis=(self.aw/(1204*eccentricity(co,si)))[...,None]*(co[...,None]*INVERSE[:,0]+si[...,None]*INVERSE[:,1])
        response=self.rw*t[...,None]+c[...,None]*axis
        safe=np.isfinite(response).all(axis=-1)&np.all(abs(response)<400,axis=-1)
        r=np.where(safe[...,None],response,0.)
        with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
            cones=np.sign(r)*1024*np.power(abs(r)*(400-self.rw)/(self.rw*(400-abs(r))),np.float32(50/21))
            rgb=self.native(cones,self.matrix)
            top=safe&(t>.5)&np.all(response>=0,axis=-1)
            if np.any(top):
                deficit=(1-t[top,None])-(c[top,None]*axis[top])/self.rw
                cones=-1024*expm1(np.float32(50/21)*(log1p(-deficit)-log1p(self.rw/(400-self.rw)*deficit)))
                rgb[top]=1024-self.native(cones,self.matrix)
        return np.where(safe[...,None],rgb,np.nan).astype('f4')

    def _hk_export(self,state,reference):
        """Apply the historical H-K knee using carried, signed energy."""
        q=np.asarray(state,dtype=np.float32)
        if not self.hk_guard:return q
        ref=np.broadcast_to(np.asarray(reference,dtype=np.float32),q.shape)
        h=q[...,0];c=np.hypot(q[...,1],q[...,2]);carried=q[...,3]
        b=np.sqrt(66*c)
        u=divide(carried,np.hypot(np.sqrt(abs(carried)),b)+b)
        # D=H²-66C agrees with carried E for u>=0; otherwise D=E+2u².
        d=np.where(carried>=0,carried,carried+2*u*u)
        e0=np.maximum(ref[...,3],0.);h2=h*h
        low=np.minimum(e0,h2); high=np.maximum(e0,h2)
        delta=divide(low,1+divide(low,high))
        active=np.isfinite(q).all(axis=-1)&np.isfinite(ref).all(axis=-1)&(h>=0)&(d<delta)
        if not np.any(active):return q
        # v=1/(1+t) bounds the fourth-root polynomial even far beyond the
        # boundary. Delta=0 supplies its limiting hard repair E=max(D,0).
        v=divide(delta,delta+np.maximum(delta-d,0.));w=1-v
        e=delta*np.sqrt(v)/np.sqrt(np.sqrt(v*v+4*v*w+10*w*w))
        z=np.maximum(e-d,0.);l=np.float32(.44)*h2
        g=l*divide(z,np.hypot(2*l,z))
        x=h2+g;cg=np.minimum(c,np.maximum(0.,(x-e)/66.))
        result=q.copy();result[...,0]=np.sqrt(np.maximum(x,0.))
        result[...,1:3]*=divide(cg,c)[...,None];result[...,3]=e
        return np.where(active[...,None],result,q)

    def export(self,state,reference=None):
        """Preserve J_A/hue and emit the last RGB tested inside AP1 [0,1024]."""
        q=np.asarray(state,dtype=np.float32)
        if reference is not None:q=self._hk_export(q,reference)
        shape=q.shape[:-1];rows=q.reshape(-1,4)
        out=np.zeros((len(rows),3),dtype=np.float32);e=rows[:,3]
        tip=e>=self.white*self.white;out[tip]=1024;active=(e>0)&~tip
        if not np.any(active):return out.reshape(*shape,3)
        ids=np.flatnonzero(active);source=rows[ids];direct=self.decode(source)
        valid=np.isfinite(direct).all(axis=-1)&np.all((direct>=0)&(direct<=1024),axis=-1)
        out[ids[valid]]=direct[valid];ids=ids[~valid];source=source[~valid]
        if not len(ids):return out.reshape(*shape,3)
        c=np.hypot(source[:,1],source[:,2]);direction=divide(source[:,1:3],c[:,None])
        ja=np.sqrt(source[:,3]);t=np.power(ja/self.white,1/GAMMA)
        co,si=direction.T
        axis=(self.aw/(1204*eccentricity(co,si)))[:,None]*(co[:,None]*INVERSE[:,0]+si[:,None]*INVERSE[:,1])
        bound=np.where(axis>0,divide(self.response_limits[1]-self.rw*t[:,None],axis),
                       np.where(axis<0,divide(self.response_limits[0]-self.rw*t[:,None],axis),np.inf))
        high=np.minimum(c,bound.min(axis=-1)).astype('f4');low=np.zeros_like(c)
        def candidate(radius):
            trial=np.column_stack((np.sqrt(source[:,3]+66*radius),direction*radius[:,None],source[:,3])).astype('f4')
            rgb=self.decode(trial)
            legal=np.isfinite(rgb).all(axis=-1)&np.all((rgb>=0)&(rgb<=1024),axis=-1)
            return rgb,legal
        saved,legal=candidate(low)
        if not np.all(legal):raise ArithmeticError('Interior cube neutral failed decoder validation')
        seed=self.capacity(source[:,3],source[:,1:3]);use=(seed>0)&(seed<high)&np.isfinite(seed)
        rgb,legal=candidate(np.where(use,seed,low));accept=use&legal
        low=np.where(accept,seed,low);saved=np.where(accept[:,None],rgb,saved)
        high=np.where(use&~legal,seed,high)
        rgb,legal=candidate(high);low=np.where(legal,high,low);saved=np.where(legal[:,None],rgb,saved)
        for _ in range(24):
            middle=low+(high-low)*np.float32(.5);moving=(middle!=low)&(middle!=high)
            if not np.any(moving):break
            rgb,legal=candidate(middle);accept=moving&legal
            low=np.where(accept,middle,low);saved=np.where(accept[:,None],rgb,saved)
            high=np.where(moving&~legal,middle,high)
        out[ids]=saved
        return out.reshape(*shape,3)

    def evaluate(self,rgb):
        if self.identity:
            return np.asarray(rgb,dtype=np.float32).copy()
        toned=self.tone_state(self.enter(rgb))
        return self.export(self.grade(toned),toned)
