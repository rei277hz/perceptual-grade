"""Deterministic robust 2-D location/scatter for scopes and white-point dragging.

Six DetMCD starts and C-steps follow Hubert, Rousseeuw & Verdonck (2012),
A deterministic algorithm for robust location and scatter, JCGS 21:618–637.
No random starts and no FastMCD dependency. See NOTICE for algorithm attribution.
The bounded scope population is converted to an equal-weight, hash-selected
sample before fitting. An ellipse is a nominal Gaussian data contour, not a confidence
interval for the mean. Floors below affect the display, never graded pixels.
"""
from statistics import NormalDist
from functools import lru_cache
from operator import index as integer_index
import numpy as np

CHI2_95=5.991464547107979
CHI2_975=7.3777589082278725
NORMAL=NormalDist()
QN_NORMAL=1/(2**.5*NORMAL.inv_cdf(.625))


def weighted_sample(points,weights,limit=512):
    """Midpoint-CDF resampling; pixel order and weights fix all chosen identities."""
    x=np.asarray(points,float).reshape(-1,2);w=np.asarray(weights,float).ravel()
    valid=np.isfinite(x).all(axis=1)&np.isfinite(w)&(w>0)
    ids=np.flatnonzero(valid);w=w[valid]
    if not len(ids):return x[:0],ids
    n=min(limit,len(ids));cdf=np.cumsum(w);cdf/=cdf[-1]
    chosen=ids[np.searchsorted(cdf,(np.arange(n)+.5)/n,side='left')]
    return x[chosen],chosen


def _pixel_hash(keys):
    """SplitMix64 permutation of stable raster IDs, with no frame/random seed."""
    z=np.asarray(keys,dtype=np.uint64)+np.uint64(0x9E3779B97F4A7C15)
    z=(z^(z>>30))*np.uint64(0xBF58476D1CE4E5B9)
    z=(z^(z>>27))*np.uint64(0x94D049BB133111EB)
    return z^(z>>31)


def hashed_sample(points,weights,limit=512,keys=None):
    """Distinct opacity-biased scope pixels, returned in stable raster-ID order.

    Fixed hash priorities scatter samples without raster-periodic strides.
    Alpha weights exponential priorities, not repeated sample entries; when
    all eligible pixels fit, every positive-alpha pixel is retained once.
    ``keys`` are original raster IDs when points were compacted into a tone.
    """
    x=np.asarray(points,float).reshape(-1,2);w=np.asarray(weights,float).ravel()
    if len(w)!=len(x):raise ValueError('Scope sample weights must match points')
    if isinstance(limit,(bool,np.bool_)):raise ValueError('Scope sample limit must be a nonnegative integer')
    try:limit=integer_index(limit)
    except TypeError:raise ValueError('Scope sample limit must be a nonnegative integer') from None
    if limit<0:raise ValueError('Scope sample limit must be a nonnegative integer')
    if keys is None:keys=np.arange(len(x),dtype=np.uint64)
    else:
        keys=np.asarray(keys)
        if keys.ndim!=1 or len(keys)!=len(x) or keys.dtype.kind not in 'iu':
            raise ValueError('Scope sample keys must be one integer raster ID per point')
        if keys.dtype.kind=='i' and np.any(keys<0):raise ValueError('Scope sample keys must be nonnegative')
        keys=keys.astype(np.uint64,copy=False)
        if len(np.unique(keys))!=len(keys):raise ValueError('Scope sample keys must be unique')
    valid=np.isfinite(x).all(axis=1)&np.isfinite(w)&(w>0)
    ids=np.flatnonzero(valid);n=min(limit,len(ids))
    if not n:return x[:0],ids[:0]
    selected_keys=keys[ids]
    if n<len(ids):
        hashes=_pixel_hash(selected_keys);alpha=w[ids]
        if np.all(alpha==alpha[0]):priority=~hashes
        else:
            # 52-bit midpoints remain strictly inside (0,1) in binary64.
            u=((hashes>>12).astype(float)+.5)*2.**-52
            # log-domain exponential race avoids overflow for tiny alpha.
            priority=np.log(-np.log(u))-np.log(alpha)
        boundary=np.partition(priority,n-1)[n-1]
        below=np.flatnonzero(priority<boundary);ties=np.flatnonzero(priority==boundary)
        ties=ties[np.argsort(selected_keys[ties],kind='stable')[:n-len(below)]]
        chosen=np.r_[below,ties]
        ids=ids[chosen];selected_keys=selected_keys[chosen]
    ids=ids[np.argsort(selected_keys,kind='stable')]
    return x[ids],ids


@lru_cache(maxsize=8)
def _qn_indices(n):
    result=np.triu_indices(n,1)
    for indices in result:indices.flags.writeable=False
    return result


def qn(x):
    x=np.asarray(x,float);n=len(x)
    if n<2:return 0.
    i,j=_qn_indices(n);d=x[i]-x[j];np.abs(d,out=d)
    h=n//2+1;k=h*(h-1)//2-1;d.partition(k)
    return QN_NORMAL*float(d[k])


def scatter(x):
    mu=np.mean(x,axis=0);z=x-mu
    return mu,z.T@z/len(x)


def regular(cov,relative=1e-10,absolute=1e-14):
    cov=(np.asarray(cov)+np.asarray(cov).T)*.5
    values,vectors=np.linalg.eigh(cov)
    floor=max(absolute,relative*max(float(values[-1]),0))
    return (vectors*np.maximum(values,floor))@vectors.T


def distances(x,mu,cov):
    z=x-mu
    return np.einsum('ni,ij,nj->n',z,np.linalg.inv(regular(cov)),z)


def ranks(x):
    order=np.argsort(x,kind='stable');v=x[order];result=np.empty(len(x),float)
    starts=np.r_[0,np.flatnonzero(np.diff(v))+1];ends=np.r_[starts[1:],len(v)]
    result[order]=np.repeat((starts+ends+1)/2,ends-starts)
    return result


@lru_cache(maxsize=8)
def _normal_scores(n):
    """Exact scalar normal scores for every possible half-integer tied rank."""
    result=np.fromiter((NORMAL.inv_cdf((rank/2-1/3)/(n+1/3))
                        for rank in range(1,2*n+1)),float,count=2*n)
    result.flags.writeable=False
    return result


def correlation(x):
    _,s=scatter(x);sd=np.sqrt(np.maximum(np.diag(s),1e-20))
    return regular(s/np.outer(sd,sd))


def starts(z):
    n=len(z);h=(n+1)//2
    yield correlation(np.tanh(z))
    r=np.column_stack([ranks(z[:,i]) for i in range(2)])
    yield correlation(r)
    normal=_normal_scores(n)[(2*r).astype(np.intp)-1]
    yield correlation(normal)
    lengths=np.linalg.norm(z,axis=1)
    signs=np.divide(z,lengths[:,None],out=np.zeros_like(z),where=lengths[:,None]>0)
    yield regular(signs.T@signs/n)
    yield regular(scatter(z[np.argsort(lengths,kind='stable')[:h]])[1])
    # Raw orthogonalized Gnanadesikan–Kettenring initial scatter.
    v=(qn(z[:,0]+z[:,1])**2-qn(z[:,0]-z[:,1])**2)/4
    ogk=np.array([[qn(z[:,0])**2,v],[v,qn(z[:,1])**2]])
    _,e=np.linalg.eigh(ogk)
    yield regular((e*np.array([qn((z@e)[:,i])**2 for i in range(2)]))@e.T)


def corrected_start(z,s):
    _,e=np.linalg.eigh(s);r=z@e
    scales=np.maximum([qn(r[:,i]) for i in range(2)],1e-8)
    root=(e*scales)@e.T;invroot=(e*(1/scales))@e.T
    mu=np.median(z@invroot,axis=0)@root
    return mu,regular(root@root)


def concentrate(z,mu,cov,h,checkpoint=None):
    previous=None
    for _ in range(80):
        if checkpoint is not None:checkpoint()
        d=distances(z,mu,cov);subset=np.sort(np.argsort(d,kind='stable')[:h])
        if previous is not None and np.array_equal(previous,subset):break
        previous=subset;mu,cov=scatter(z[subset]);cov=regular(cov)
    return mu,cov


def gaussian_correction(alpha):
    q=-2*np.log1p(-alpha)
    return alpha/(1-(1+q/2)*(1-alpha))


def detmcd(points,support=.75,checkpoint=None):
    """DetMCD, p=2; deterministic six starts, raw h-fit, corrected reweighting.

    Uses asymptotic Qn/Gaussian consistency corrections. Small/degenerate
    populations use their finite scatter with regularization, never random fit.
    """
    if checkpoint is not None:checkpoint()
    x=np.asarray(points,float).reshape(-1,2)
    x=x[np.isfinite(x).all(axis=1)]
    if not len(x):return np.zeros(2),np.zeros((2,2))
    # Canonical order makes tie handling invariant to input permutation.
    x=x[np.lexsort((x[:,1],x[:,0]))]
    center=np.median(x,axis=0);scale=np.array([qn(x[:,i]) for i in range(2)])
    if len(x)<6 or np.any(scale<1e-10) or np.linalg.matrix_rank(x-center)<2:
        return scatter(x)
    z=(x-center)/scale;n=len(z);h=max((n+3)//2,min(n-1,int(np.ceil(support*n))))
    fits=[]
    for initial in starts(z):
        if checkpoint is not None:checkpoint()
        mu,cov=corrected_start(z,initial)
        half=np.argsort(distances(z,mu,cov),kind='stable')[:(n+1)//2]
        mu,cov=scatter(z[half]);mu,cov=concentrate(z,mu,regular(cov),h,checkpoint)
        fits.append((np.linalg.slogdet(cov)[1],mu,cov))
    _,mu,cov=min(fits,key=lambda f:f[0]);cov*=gaussian_correction(h/n)
    keep=distances(z,mu,cov)<=CHI2_975
    if np.count_nonzero(keep)>=3:
        mu,cov=scatter(z[keep]);cov*=gaussian_correction(.975)
    return mu*scale+center,cov*np.outer(scale,scale)


def describe_sample(points,min_std=.05,checkpoint=None):
    sample=np.asarray(points,float).reshape(-1,2)
    sample=sample[np.isfinite(sample).all(axis=1)]
    center,cov=detmcd(sample,checkpoint=checkpoint)
    values,vectors=np.linalg.eigh((cov+cov.T)*.5)
    # A finite full data ellipse is always available, including rank 0/1.
    display_cov=(vectors*np.maximum(values,min_std**2))@vectors.T
    return dict(center=center,covariance=cov,ellipse_covariance=display_cov,
                count=len(sample),reference=not len(sample))


def describe(points,weights,limit=512,min_std=.05):
    sample,ids=weighted_sample(points,weights,limit)
    result=describe_sample(sample,min_std);result['sample_indices']=ids
    return result


def ellipse(stats,count=96,min_radius=0):
    values,vectors=np.linalg.eigh(stats['ellipse_covariance'])
    radii=np.maximum(np.sqrt(CHI2_95*np.maximum(values,0)),min_radius)
    t=np.linspace(0,2*np.pi,count,endpoint=False)
    return stats['center']+np.column_stack([np.cos(t),np.sin(t)])@(vectors*radii).T
