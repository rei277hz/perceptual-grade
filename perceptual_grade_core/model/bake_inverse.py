"""Portable float32 Stack inverse; the same finite search is implemented in GLSL.

This is a found-root inverse, not an inverse of the final cube guard.
The independent float64 least-squares oracle remains in stack.py. Numerical
search is confined to LUT construction; boundary lookup never solves a root.
"""
import numpy as np
from .bake import divide

TOLERANCE=np.float32(1e-4)  # Raw appearance acceptance, independent of search stopping.
RGB_RELATIVE_TOLERANCE=np.float32(.0001)
RGB_ABSOLUTE_TOLERANCE=np.float32(.00003)
SEARCH_TOLERANCE=np.float32(1e-5)
CHROMA_TOE_MAX=np.float32(2)
CHROMA_TOE_MIN=np.float32(2**-126)
STEP=np.float32(.002)
ITERATIONS=8


def features(q):
    return np.stack((q[...,0],np.sign(q[...,3])*np.sqrt(abs(q[...,3])),q[...,1],q[...,2]),-1)


def chroma_scale(target):
    """Fixed per-target toe width, including signed and zero-height requests."""
    cap=np.float32(66)*CHROMA_TOE_MAX
    height=np.minimum(abs(target[...,0]),np.sqrt(cap))
    energy=np.maximum(height*height,np.minimum(abs(target[...,3]),cap))
    ab=np.minimum(abs(target[...,1:3]),CHROMA_TOE_MAX)
    radius=np.hypot(ab[...,0],ab[...,1])
    return np.maximum(CHROMA_TOE_MIN,np.minimum(CHROMA_TOE_MAX,np.maximum(energy/66,radius)))


def chroma_encode(log_value,log_scale):
    """C2 linear/quadratic toe joined to log chroma; log(0) is -infinity."""
    ell=np.asarray(log_value,dtype='f4')-log_scale
    u=np.exp(np.minimum(ell,0))
    return np.where(ell<=0,u*(2-.5*u),1.5+ell).astype('f4')


def chroma_decode(coordinate,log_scale):
    """Decode only the selected C2 branch, retaining exact chroma zero."""
    x,logtau=np.broadcast_arrays(coordinate,log_scale)
    low=x<=1.5;result=np.empty_like(x,dtype='f4')
    v=x[low]
    result[low]=np.exp(logtau[low])*(2*v/(2+np.sqrt(np.maximum(0,4-2*v))))
    # Form the exponent before indexing to preserve Python-scalar promotion.
    result[~low]=np.exp(np.asarray(coordinate-1.5+log_scale)[~low])
    return result


def opponent_encode(ab,log_scale):
    """Cartesian radial toe/log coordinates; exact zero has no hue singularity."""
    ab=np.asarray(ab,dtype='f4');c=np.hypot(ab[...,0],ab[...,1])
    logc=np.log(c,out=np.full_like(c,-np.inf),where=c>0)
    radius=chroma_encode(logc,log_scale)
    return ab*divide(radius,c)[...,None]


def opponent_decode(x,log_scale):
    x=np.asarray(x,dtype='f4');radius=np.hypot(x[...,0],x[...,1])
    c=chroma_decode(radius,log_scale)
    return x*divide(c,radius)[...,None]


def normal_step(jac,residual,damping):
    """Explicit damped Cholesky solve for two or three residual coordinates."""
    dimension=jac.shape[-1]
    matrix=np.empty((len(jac),dimension,dimension),dtype='f4')
    gradient=np.empty((len(jac),dimension),dtype='f4')
    for i in range(dimension):
        gradient[:,i]=np.sum(jac[:,:,i]*residual,axis=-1)
        for j in range(dimension):matrix[:,i,j]=np.sum(jac[:,:,i]*jac[:,:,j],axis=-1)
    lam=damping*(1+np.trace(matrix,axis1=1,axis2=2))
    lower=np.zeros_like(matrix)
    for i in range(dimension):
        for j in range(i+1):
            value=matrix[:,i,j]-np.sum(lower[:,i,:j]*lower[:,j,:j],axis=-1)
            lower[:,i,j]=np.sqrt(np.maximum(value+lam,lam)) if i==j else value/lower[:,j,j]
    y=np.zeros_like(gradient);step=np.zeros_like(gradient)
    for i in range(dimension):y[:,i]=(gradient[:,i]-np.sum(lower[:,i,:i]*y[:,:i],axis=-1))/lower[:,i,i]
    for i in range(dimension-1,-1,-1):step[:,i]=(y[:,i]-np.sum(lower[:,i+1:,i]*step[:,i+1:],axis=-1))/lower[:,i,i]
    return step


def minimize(residual,seed,low,high,iterations,probe_batch=True):
    """Finite damped Broyden with independent lane recovery and line search."""
    seed=np.asarray(seed,dtype='f4')
    low=np.broadcast_to(np.asarray(low,dtype='f4'),seed.shape)
    high=np.broadcast_to(np.asarray(high,dtype='f4'),seed.shape)
    x=np.clip(seed,low,high);n,dimension=x.shape
    ids=np.arange(n);r=residual(x,ids);score=np.sum(r*r,axis=-1)
    jac=np.zeros((n,r.shape[-1],dimension),dtype='f4');damping=np.full(n,1e-4,dtype='f4')
    fresh=np.ones(n,dtype=bool);measured=np.zeros(n,dtype=bool)
    offsets=np.repeat(np.eye(dimension,dtype='f4'),2,axis=0)*np.tile([STEP,-STEP],dimension)[:,None]
    for iteration in range(iterations):
        remaining=r if iteration==0 else r[ids]
        ids=ids[np.max(abs(remaining),axis=-1)>SEARCH_TOLERANCE]
        if not len(ids):break
        refresh=ids[fresh[ids]|((iteration%12==0)&~measured[ids])]
        if len(refresh):
            v=x[refresh]
            if probe_batch:
                probes=np.clip(v[None,:,:]+offsets[:,None,:],low[refresh],high[refresh])
                values=residual(probes.reshape(-1,dimension),np.tile(refresh,2*dimension)).reshape(2*dimension,len(v),r.shape[-1])
                for axis in range(dimension):
                    hi=2*axis;lo=hi+1
                    jac[refresh,:,axis]=(values[hi]-values[lo])/(probes[hi,:,axis]-probes[lo,:,axis])[:,None]
            else:
                for axis in range(dimension):
                    hi=np.clip(v+offsets[2*axis],low[refresh],high[refresh]);lo=np.clip(v+offsets[2*axis+1],low[refresh],high[refresh])
                    jac[refresh,:,axis]=(residual(hi,refresh)-residual(lo,refresh))/(hi[:,axis]-lo[:,axis])[:,None]
            measured[refresh]=True
        v=x[ids];base=r[ids];j=jac[ids]
        step=normal_step(j,base,damping[ids]);step/=np.maximum(1,np.max(abs(step),axis=-1)/1.5)[:,None]
        improved=np.zeros(len(ids),dtype=bool)
        for alpha in (1.,.5,.125):
            rem=np.flatnonzero(~improved)
            if not len(rem):break
            ri=ids[rem];candidate=np.clip(v[rem]-np.float32(alpha)*step[rem],low[ri],high[ri])
            rr=residual(candidate,ri);ss=np.sum(rr*rr,axis=-1)
            good=ss<score[ri];picked=ri[good];ii=rem[good]
            delta=candidate[good]-v[ii];jj=j[ii]
            change=rr[good]-base[ii]-np.sum(jj*delta[:,None,:],axis=-1)
            square=np.sum(delta*delta,axis=-1)
            direction=np.divide(delta,square[:,None],out=np.zeros_like(delta),where=square[:,None]>1e-20)
            jac[picked]=j[ii]+change[:,:,None]*direction[:,None,:]
            measured[picked]=False;x[picked]=candidate[good];r[picked]=rr[good];score[picked]=ss[good];improved[ii]=True
        fresh[ids]=~improved&~measured[ids]
        damping[ids]=np.where(improved,np.maximum(1e-6,damping[ids]*.2),np.minimum(1e6,damping[ids]*10))
    return x


class Inverse:
    def __init__(self,kernel,canonical,weights):
        self.kernel=kernel;self.canonical=np.asarray(canonical,dtype='f4');self.weights=np.asarray(weights,dtype='f4')
        p=kernel.color
        self.simple=not kernel.web_active and not any(np.any(getattr(p,n)) for n in ('lightness','chroma','adaptance'))
        self.probe_batch=True;self.tone_table=None

    def repair(self,values):
        x=np.asarray(values,dtype='f4').copy();active=np.any((x<0)|(x>1024),axis=-1)
        if not np.any(active):return x
        v=x[active];amplitude=np.max(abs(v),axis=-1);unit=v/amplitude[:,None]
        y=np.maximum(0,np.sum(unit*self.weights,axis=-1));k=np.maximum(0,-np.min(unit,axis=-1))
        repaired=(unit+k[:,None])*divide(y,y+k)[:,None]
        unit=np.where((k>0)[:,None],repaired,unit);peak=np.max(unit,axis=-1)
        scale=np.minimum(amplitude,np.nextafter(divide(np.float32(1024),peak),np.float32(-np.inf)))
        x[active]=unit*np.maximum(0,scale)[:,None]
        return x

    def undo_tone(self,q):
        k=self.kernel;tone=k.tone
        if not tone.active:return q.copy()
        j=np.sqrt(np.maximum(q[:,3],0));c=np.hypot(q[:,1],q[:,2]);h=np.sqrt(j*j+66*c)
        if self.tone_table is None:raise RuntimeError('Prepare the inverse tone table before evaluating pixels')
        t=self.tone_table.evaluate(h)
        original=np.where(h==0,0,100*(np.float32(tone.black)+np.float32(tone.span)*t))
        gain=divide(2*original*original,np.hypot(66*c,2*j*original)+66*c)
        jj=j*gain;ab=q[:,1:3]*gain[:,None]
        return np.column_stack((np.sqrt(jj*jj+66*np.hypot(ab[:,0],ab[:,1])),ab,jj*jj)).astype('f4')

    def direct_state(self,raw):
        """Undo the simple creative family; retain carried energy for tone alone."""
        q=raw.copy()
        if self.kernel.color.global_gain!=0:
            q[:,1:3]*=np.exp2(np.float32(-self.kernel.color.global_gain))
            c=np.hypot(q[:,1],q[:,2]);q[:,3]=np.maximum(0,q[:,0]*q[:,0]-66*c)
            q[:,0]=np.sqrt(q[:,3]+66*c)
        return self.undo_tone(q)

    def feasible(self,q):
        """Test actual decoded RGB; fitted capacity does not certify legality."""
        rgb=self.kernel.decode(q)
        return np.isfinite(rgb).all(axis=-1)&np.all((rgb>=0)&(rgb<=1024),axis=-1)

    def reduced(self,x,target,output=False,scale=None):
        """Three-variable raw solve: log J_A and Cartesian opponent coordinates.

        Candidate grading evaluates the complete web with frozen pre-grade tone.
        All four raw features, including carried signed energy, are fitted.
        """
        if scale is None:
            tau=chroma_scale(target);scale=(tau,np.log(tau))
        _,logtau=scale
        j=np.exp(x[:,0]);ab=opponent_decode(x[:,1:3],logtau);c=np.hypot(ab[:,0],ab[:,1])
        state=np.column_stack((np.sqrt(j*j+66*c),ab,j*j)).astype('f4')
        if output:return self.undo_tone(state)
        actual=self.kernel.grade(state)
        opponents=opponent_encode(actual[:,1:3],logtau)-opponent_encode(target[:,1:3],logtau)
        ja=np.sign(actual[:,3])*np.sqrt(abs(actual[:,3]));wanted=np.sign(target[:,3])*np.sqrt(abs(target[:,3]))
        return np.column_stack(((actual[:,0]-target[:,0])/(1+abs(target[:,0])),
                                (ja-wanted)/(1+abs(wanted)),opponents)).astype('f4')

    def acceptance_limit(self,target,appearance=False):
        """RGB absolute/relative acceptance, expressed in reported residual units."""
        if appearance:return np.full(target.shape[:-1],TOLERANCE,dtype='f4')
        compared=target if self.kernel.identity else self.kernel.native(target,self.canonical)
        peak=np.max(abs(compared),axis=-1)
        return np.maximum(RGB_RELATIVE_TOLERANCE*peak,RGB_ABSOLUTE_TOLERANCE)/np.maximum(peak,np.float32(1e-4))

    def validation_target(self,target,appearance):
        if appearance:
            wanted=features(target)
            return wanted,1+abs(wanted)
        compared=target if self.kernel.identity else self.kernel.native(target,self.canonical)
        return target,np.maximum(np.max(abs(compared),axis=-1),np.float32(1e-4))[:,None]

    def forward_prior(self,rgb,appearance):
        k=self.kernel;entered=k.enter(rgb);toned=k.tone_state(entered)
        actual=k.grade(toned)
        return entered,toned,actual if appearance else k.export(actual,toned)

    def residual(self,rgb,target,appearance,prepared=None,output=None):
        k=self.kernel
        wanted,divisor=self.validation_target(target,appearance) if prepared is None else prepared
        if k.identity and not appearance:return np.column_stack(((rgb-target)/divisor,np.zeros(len(rgb),dtype='f4')))
        if output is None:
            toned=k.tone_state(k.enter(rgb));actual=k.grade(toned)
            if not appearance:actual=k.export(actual,toned)
        else:actual=output
        if appearance:return (features(actual)-wanted)/divisor
        delta=k.native(actual-target,self.canonical)
        return np.column_stack((delta/divisor,np.zeros(len(rgb),dtype='f4')))

    def evaluate(self,target,appearance=False):
        """Return RGB and maximum normalized reproduction residual per pixel."""
        q=np.asarray(target,dtype='f4');width=4 if appearance else 3
        if q.shape[-1:]!=(width,) or not np.isfinite(q).all():raise ValueError('Invalid inverse input shape or nonfinite value')
        shape=q.shape[:-1];target=q.reshape(-1,width)
        if not appearance:target=self.repair(target)
        k=self.kernel;prior=np.clip(k.export(target),0,1024) if appearance else target.copy()
        if not len(prior):return np.empty((*shape,4),dtype='f4')
        prepared=self.validation_target(target,appearance)
        entered,toned,output=self.forward_prior(prior,appearance) if not k.identity or appearance else (None,None,None)
        best=prior.copy();error=np.max(abs(self.residual(best,target,appearance,prepared,output)),axis=1)
        if k.identity:return np.column_stack((best,error)).reshape(*shape,4)
        # A root at the identity prior has the globally smallest ranking distance.
        limit=self.acceptance_limit(target,appearance)
        result=np.column_stack((best,error));active=np.flatnonzero(error>limit)
        if not len(active):return result.reshape(*shape,4)
        target=target[active];prior=prior[active];best=best[active];error=error[active];limit=limit[active]
        prepared=tuple(value[active] for value in prepared)
        canonical_prior=np.arcsinh(k.native(prior,self.canonical));distance=np.zeros(len(prior),dtype='f4')
        canonical_best=k.native(best,self.canonical)
        def choose(candidate,rows,output=None):
            candidate=np.clip(candidate,0,1024)
            rr=np.max(abs(self.residual(candidate,target[rows],appearance,tuple(value[rows] for value in prepared),output)),axis=1)
            canonical=k.native(candidate,self.canonical)
            dd=np.sum((np.arcsinh(canonical)-canonical_prior[rows])**2,axis=1)
            old=error[rows];root=rr<=limit[rows];oldroot=old<=limit[rows]
            lex=(canonical[:,0]<canonical_best[rows,0])|((canonical[:,0]==canonical_best[rows,0])&
                ((canonical[:,1]<canonical_best[rows,1])|((canonical[:,1]==canonical_best[rows,1])&(canonical[:,2]<canonical_best[rows,2]))))
            near=(dd<distance[rows])|((dd==distance[rows])&lex)
            pick=(root&~oldroot)|((root==oldroot)&np.where(root,near,(rr<old)|((rr==old)&near)))
            ids=rows[pick];best[ids]=candidate[pick];error[ids]=rr[pick];distance[ids]=dd[pick];canonical_best[ids]=canonical[pick]
        rows=np.arange(len(prior));choose(np.zeros_like(prior),rows,np.zeros(width,dtype='f4'))
        raw=target if appearance else entered[active]
        if self.simple:
            direct=self.direct_state(raw);regular=np.clip(k.export(direct),0,1024);choose(regular,rows)
        else:
            base=toned[active]
            tau=chroma_scale(raw);logtau=np.log(tau)
            ab=base[:,1:3]*np.exp2(np.float32(-k.color.global_gain))
            seed=np.column_stack((np.log(np.maximum(np.sqrt(np.maximum(base[:,3],0)),np.float32(1e-30))),
                opponent_encode(ab,logtau)))
            bound=chroma_encode(np.float32(20),logtau)
            low=np.column_stack((np.full_like(tau,-69),-bound,-bound))
            high=np.column_stack((np.full_like(tau,20),bound,bound))
            fit=minimize(lambda x,ids:self.reduced(x,raw[ids],scale=(tau[ids],logtau[ids])),seed,
                low,high,ITERATIONS,self.probe_batch)
            choose(k.export(self.reduced(fit,raw,True,(tau,logtau))),rows)
        # Candidate generation uses only the raw inverse.  RGB candidates are
        # still ranked by complete guarded forward reproduction; return an
        # honest approximation when none meets acceptance, without a second fit.
        if not np.isfinite(best).all() or not np.isfinite(error).all():raise ArithmeticError('Nonfinite inverse search result')
        result[active]=np.column_stack((best,error))
        return result.reshape(*shape,4)
