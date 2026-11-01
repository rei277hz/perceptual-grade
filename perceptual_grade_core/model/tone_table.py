"""Prepared C2 scalar tone inverse; pixel evaluation never solves an equation.

The table represents t(H)/H in log1p(H/scale), retaining relative precision
at black. Shared nodal derivatives define quintic Hermite cells. The positive
linear tail has an explicit bound on the discarded sigmoid remainder. Its
endpoint jets agree to the precision of the stored float32 coefficients.
"""
from math import comb
import numpy as np
from .bake import log1p

MAX_CELLS=2048
MAX_ROUNDS=32
MAX_ROOT_EVALUATIONS=80
MAX_BYTES=64*1024
_GL_X,_GL_W=np.polynomial.legendre.leggauss(8)
_GL_X=(_GL_X+1)*.5
_GL_W=_GL_W*.5
_BERNSTEIN=np.array([[comb(k,j)/comb(5,j) if j<=k else 0 for j in range(6)] for k in range(6)])


def hermite(left,right,u):
    """Shared factorization, also used by the GLSL scalar table lookup."""
    width=right[...,0]-left[...,0]
    z=np.clip((u-left[...,0])/width,0,1);a=1-z
    z2=z*z;z3=z2*z;a2=a*a;a3=a2*a
    blend=z3*(10+z*(-15+6*z))
    first=left[...,2]*z*a3*(1+3*z)-right[...,2]*z3*a*(4-3*z)
    second=left[...,3]*z2*a3+right[...,3]*z3*a2
    return left[...,1]+(right[...,1]-left[...,1])*blend+width*first+width*width*.5*second


def _monotone(nodes):
    """Conservative Bernstein bounds for q>0 and d(H*q)/dH>0.

    Subdivision tightens interval bounds without changing the stored curve.
    Float32 evaluation rounding is checked separately against the oracle.
    """
    left,right=nodes[:-1].astype(float),nodes[1:].astype(float)
    width=right[:,0]-left[:,0];delta=right[:,1]-left[:,1]
    a=width*left[:,2];b=width*right[:,2];c=width*width*left[:,3];d=width*width*right[:,3]
    powers=np.column_stack((left[:,1],a,.5*c,10*delta-6*a-4*b-1.5*c+.5*d,
        -15*delta+8*a+7*b+1.5*c-d,6*delta-3*a-3*b-.5*c+.5*d))
    coefficients=powers@_BERNSTEIN.T;low=left[:,0];high=right[:,0]
    ids=np.arange(len(width));good=np.ones(len(width),dtype=bool)
    for depth in range(9):
        derivative=5*np.diff(coefficients,axis=1)/(high-low)[:,None]
        minimum=derivative.min(axis=1);r0=-np.expm1(-low);r1=-np.expm1(-high)
        bound=coefficients.min(axis=1)+np.minimum(r0*minimum,r1*minimum)
        unresolved=(coefficients.min(axis=1)<=0)|(bound<=0)
        if not np.any(unresolved):break
        ids=ids[unresolved];coefficients=coefficients[unresolved];low=low[unresolved];high=high[unresolved]
        if depth==8:
            good[ids]=False;break
        layers=[coefficients]
        for _ in range(5):layers.append((layers[-1][:,:-1]+layers[-1][:,1:])*.5)
        first=np.column_stack([v[:,0] for v in layers]);last=np.column_stack([v[:,-1] for v in layers[::-1]])
        middle=(low+high)*.5
        coefficients=np.concatenate((first,last));ids=np.tile(ids,2)
        low,high=np.concatenate((low,middle)),np.concatenate((middle,high))
    return good


class _Curve:
    def __init__(self,tone):
        self.knots=np.asarray(tone.knots,dtype=float);self.width=float(tone.width)
        self.slope=float(tone.slope);self.residual=np.asarray(tone.residual,dtype=float)
        if not (np.isfinite(self.knots).all() and np.all(np.diff(self.knots)>0) and
                np.isfinite(self.width) and self.width>0 and np.isfinite(self.slope) and self.slope>0):
            raise ValueError('Tone table requires finite increasing knots, positive width and slope')
        self.log_gap=np.log(-np.expm1(-np.diff(self.knots)/self.width))
        self.base=np.log1p(np.exp(-self.knots[:-1]/self.width))
        self.normalizers=self.integrals(1.)
        self.factors=self.residual/self.normalizers
        self.evaluations=0;self.maximum_iterations=0;self.cache={0.:0.}

    def integrals(self,t):
        t=np.asarray(t,dtype=float)[...,None];left,right=self.knots[:-1],self.knots[1:]
        with np.errstate(divide='ignore'):
            positive=((t-left)/self.width-self.base-np.logaddexp(0,(t-right)/self.width)
                +self.log_gap+np.log(-np.expm1(-t/self.width)))
        return self.width*np.logaddexp(0,positive)

    def value(self,t):
        return 100*(self.slope*t+self.integrals(t)@self.factors)

    def derivatives(self,t,order=3):
        t=np.asarray(t,dtype=float)
        x=(t[...,None]-self.knots)/self.width;e=np.exp(-abs(x))
        s=np.where(x>=0,1/(1+e),e/(1+e));difference=lambda v:(v[...,:-1]-v[...,1:])@self.factors
        first=100*(self.slope+difference(s))
        if order==1:return first
        a=s*(1-s)
        return first,100*difference(a)/self.width,100*difference(a*(1-2*s))/(self.width*self.width)

    def inverse(self,h):
        shape=np.shape(h);h=np.asarray(h,dtype=float).reshape(-1)
        missing=np.unique([v for v in h if float(v) not in self.cache])
        if len(missing):
            low=np.zeros_like(missing);high=missing/(100*self.slope);x=(low+high)*.5
            ids=np.arange(len(x));counts=np.zeros(len(x),dtype=int)
            for iteration in range(MAX_ROOT_EVALUATIONS):
                if not len(ids):break
                xx=x[ids];value=self.value(xx);below=value<missing[ids]
                self.evaluations+=len(ids);counts[ids]+=1
                low[ids]=np.where(below,xx,low[ids]);high[ids]=np.where(below,high[ids],xx)
                middle=(low[ids]+high[ids])*.5
                done=(middle==low[ids])|(middle==high[ids])
                pending=ids[~done]
                if iteration<16 and len(pending):
                    xx=xx[~done];below=below[~done]
                    trial=xx-(value[~done]-missing[pending])/self.derivatives(xx,1)
                    trial=np.where(trial==xx,np.nextafter(xx,np.where(below,np.inf,-np.inf)),trial)
                    inside=(trial>low[pending])&(trial<high[pending])&np.isfinite(trial)
                    x[pending]=np.where(inside,trial,middle[~done])
                else:x[pending]=middle[~done]
                ids=pending
            if len(ids) and np.any((high[ids]-low[ids])>1e-12*np.maximum(high[ids],np.finfo(float).tiny)):
                raise ArithmeticError('Tone table preparation exhausted its scalar root budget')
            self.maximum_iterations=max(self.maximum_iterations,int(counts.max(initial=0)))
            self.cache.update(zip(map(float,missing),(low+high)*.5))
        return np.array([self.cache[float(v)] for v in h]).reshape(shape)

    def jets(self,h):
        h=np.asarray(h,dtype=float);t=self.inverse(h);q=np.empty_like(h);first=np.empty_like(h);second=np.empty_like(h)
        zero=h==0;positive=~zero
        if np.any(zero):
            a,b,c=self.derivatives(0.)
            q[zero]=1/a;first[zero]=-b/(2*a**3);second[zero]=(3*b*b-a*c)/(3*a**5)
        if np.any(positive):
            hh=h[positive];tt=t[positive];d,e,_=self.derivatives(tt)
            inverse_first=1/d;inverse_second=-e/d**3
            qq=tt/hh;f=(inverse_first-qq)/hh;s=(inverse_second-2*f)/hh
            near=tt/self.width<.125
            if np.any(near):
                # Integral moments avoid subtracting equal leading terms at black.
                _,dd,ddd=self.derivatives(tt[near,None]*_GL_X)
                b=(dd*_GL_X)@_GL_W;c=(ddd*_GL_X*_GL_X)@_GL_W
                a=hh[near]/tt[near];dn=d[near]
                f[near]=-b/(a*a*dn)
                s[near]=-c/(a*a*dn*dn)+2*b*b/(a**3*dn*dn)+b*e[near]/(a*a*dn**3)
            q[positive]=qq;first[positive]=f;second[positive]=s
        return q,first,second


class ToneTable:
    """Bounded per-tone preparation, shared float32 lookup for every pixel."""
    def __init__(self,tone,tolerance=2e-5):
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid tone table accuracy')
        curve=_Curve(tone);self.tolerance=float(tolerance)
        self.scale=np.float32(curve.value(curve.width));self.slope=np.float32(curve.slope)
        left,right=curve.knots[:-1],curve.knots[1:];w=curve.width
        limits=right-left+w*(np.log1p(np.exp(-right/w))-np.log1p(np.exp(-left/w)))
        intercept=float(limits@curve.factors);self.intercept=np.float32(intercept)
        if not np.isfinite(self.scale) or self.scale<=0 or self.slope<=0:
            raise ArithmeticError('Tone table scale is outside the float32 domain')
        for distance in (32.,40.,48.,56.,64.):
            tail_t=curve.knots[-1]+distance*w
            candidate=np.float32(100*(float(self.slope)*tail_t+float(self.intercept)))
            last=float(log1p(candidate/self.scale))
            endpoint=float(self.scale)*np.expm1(last)
            self.tail_h=np.float32(endpoint)
            if float(self.tail_h)<endpoint:
                self.tail_h=np.nextafter(self.tail_h,np.float32(np.inf))
            minimum=(endpoint/100-float(self.intercept))/float(self.slope)
            remainder=w*np.sum(curve.factors*np.exp((right-minimum)/w))
            self.tail_remainder_bound=float(remainder/curve.slope)
            relative=(abs(curve.slope-float(self.slope))+(abs(intercept-float(self.intercept))+remainder)/minimum)/curve.slope
            self.tail_relative_bound=float(relative/(1-relative)) if relative<1 else np.inf
            if self.tail_relative_bound<tolerance*.25:break
        else:raise ArithmeticError('Tone table tail cannot meet the requested accuracy')
        if not np.isfinite(self.tail_h):raise ArithmeticError('Tone table tail is outside the float32 domain')
        axis=np.unique(np.r_[0.,np.log1p(curve.value(curve.knots[1:-1])/float(self.scale)),last].astype('f4'))
        for round_index in range(MAX_ROUNDS):
            h=float(self.scale)*np.expm1(axis.astype(float));q,d,dd=curve.jets(h)
            # The upper endpoint uses the analytic tail's own derivatives.
            ht=h[-1];q[-1]=1/(100*float(self.slope))-float(self.intercept)/(float(self.slope)*ht)
            d[-1]=float(self.intercept)/(float(self.slope)*ht*ht);dd[-1]=-2*d[-1]/ht
            factor=h+float(self.scale)
            self.nodes=np.column_stack((axis,q,d*factor,dd*factor*factor+d*factor)).astype('f4')
            fractions=np.array([0.,.25,.5,.75,1.])
            probes=axis[:-1,None].astype(float)+(axis[1:]-axis[:-1])[:,None]*fractions
            values=(float(self.scale)*np.expm1(probes)).astype('f4')
            expected=curve.inverse(values.astype(float));actual=self.evaluate(values).astype(float)
            error=np.divide(abs(actual-expected),abs(expected),out=np.zeros_like(expected),where=expected!=0)
            self.max_error=float(error.max(initial=0))
            bad=(error.max(axis=1)>tolerance*.5)|~_monotone(self.nodes)
            if not np.any(bad):break
            if len(axis)-1+np.count_nonzero(bad)>MAX_CELLS:
                raise ArithmeticError('Tone table exceeds its 2048-cell accuracy budget')
            mid=((axis[:-1].astype(float)+axis[1:])*.5).astype('f4')[bad]
            if np.any((mid<=axis[:-1][bad])|(mid>=axis[1:][bad])):
                raise ArithmeticError('Tone table reached float32 address precision')
            axis=np.sort(np.concatenate((axis,mid)))
        else:raise ArithmeticError('Tone table exceeds its 32-round preparation budget')
        self.rounds=round_index+1;self.root_evaluations=curve.evaluations
        self.maximum_root_iterations=curve.maximum_iterations;self.prepared_samples=len(curve.cache)
        self.cells=len(self.nodes)-1
        self.bytes=self.nodes.nbytes+5*4;self.nbytes=self.bytes
        if self.bytes>MAX_BYTES:raise MemoryError('Tone table exceeds 64 KiB')

    def evaluate(self,h):
        with np.errstate(over='ignore'):values=np.asarray(h,dtype='f4')
        if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Tone inverse requires finite nonnegative lightness')
        shape=values.shape;values=values.reshape(-1);result=np.zeros_like(values)
        tail=values>=self.tail_h
        result[tail]=(values[tail]/100-self.intercept)/self.slope
        ids=np.flatnonzero((values>0)&~tail)
        if len(ids):
            height=values[ids];u=log1p(height/self.scale)
            # Use the stored coordinate for the mathematical join; tail_h is
            # only an upward-rounded fast path for large inputs.
            upper=u>=self.nodes[-1,0]
            if np.any(upper):
                result[ids[upper]]=(height[upper]/100-self.intercept)/self.slope
                ids=ids[~upper];height=height[~upper];u=u[~upper]
            index=np.clip(np.searchsorted(self.nodes[:,0],u,side='right')-1,0,len(self.nodes)-2)
            result[ids]=height*hermite(self.nodes[index],self.nodes[index+1],u)
        return result.reshape(shape)


GLSL=r'''
float pg_tone_table_inverse(float h) {
    if(h==0.0)return 0.0;
    if(h>=pg_param(118))return (h/100.0-pg_param(120))/pg_param(119);
    float u=pg_log1p(h/pg_param(117));int lo=0,hi=int(pg_param(121))-1;
    if(u>=pg_tone_node(hi,0))return (h/100.0-pg_param(120))/pg_param(119);
    for(int step=0;step<11;step++) {
        if(hi-lo<=1)break;
        int mid=(lo+hi)/2;
        if(pg_tone_node(mid,0)<=u)lo=mid;else hi=mid;
    }
    vec4 left,right;
    for(int i=0;i<4;i++) {left[i]=pg_tone_node(lo,i);right[i]=pg_tone_node(lo+1,i);}
    float width=right.x-left.x,z=clamp((u-left.x)/width,0.0,1.0),a=1.0-z;
    float z2=z*z,z3=z2*z,a2=a*a,a3=a2*a;
    float blend=z3*(10.0+z*(-15.0+6.0*z));
    float first=left.z*z*a3*(1.0+3.0*z)-right.z*z3*a*(4.0-3.0*z);
    float second=left.w*z2*a3+right.w*z3*a2;
    float q=left.y+(right.y-left.y)*blend+width*first+width*width*.5*second;
    return h*q;
}
'''
