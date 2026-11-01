"""Historical D65 cube/spindle mapping retained for comparisons, not grading."""
from functools import lru_cache, cached_property
from pathlib import Path
import numpy as np
from .boundary_curves import stencil,root
from ...color_core.modcam16 import GAMMA,INVERSE,eccentricity
from ...color_core.scaled import Scalar as S,norm,dot
from .boundary_cache import Boundary
from .boundary_resource import LEGACY_FORMAT as FORMAT,BUDGET,load_data
from types import MappingProxyType

MAXIMUM=1024.


def white_response(fl):
    p=np.power(np.float64(fl)*MAXIMUM,.42)
    return float(400*p/(27.13+p))


def image(values,matrix):
    return np.stack([values[...,1]+row[0]*(values[...,0]-values[...,1])+
                     row[2]*(values[...,2]-values[...,1]) for row in matrix],axis=-1)


def map_radius(radius,body,capacity_ratio,release=False):
    r,b,k=np.broadcast_arrays(np.asarray(radius,dtype=float),body,capacity_ratio)
    if np.any(r<0) or np.any(b<0) or np.any(k<1):raise ValueError('Invalid radial domain')
    end=b*(1 if release else k)
    x=np.divide(np.minimum(r,end),b,out=np.zeros_like(r),where=b>0)
    result=b*x*(1+(k-1)*x*x) if release else b*root(1.,k-1,x)
    return np.where(r>=end,b*(k if release else 1.),result)


def unit_capacity(position,angle,fl,aw,matrix,checkpoint=None):
    return context(fl,aw,matrix).unit(position,angle,checkpoint)


class Context:
    def __init__(self,knots,profile,white,fl,aw,matrix,cache=None):
        self.knots,self.profile=knots,profile
        self.cache=cache
        self.white,self.fl,self.aw,self.matrix=white,fl,aw,matrix
        for v in (knots,profile,matrix):v.setflags(write=False)
        if cache is not None:
            for v in cache.values():v.setflags(write=False)

    def coefficient(self,t):
        ids,weights=stencil(self.knots,np.asarray(t,dtype=float))
        return np.sum(self.profile[ids]*weights,axis=-1)

    @cached_property
    def lookup(self):
        from .boundary_table import Table
        if self.cache is None:raise ValueError('Missing boundary resource')
        return Table(self.cache)

    def body(self,ja):
        t=(np.asarray(ja,dtype=float)/self.white)**(1/GAMMA)
        return self.white*t*(1-t)*self.coefficient(t)

    def unit(self,t,hue,checkpoint=None):return Boundary(self).capacity(t,hue,checkpoint)

    def capacity(self,ja,hue):
        t=(np.asarray(ja,dtype=float)/self.white)**(1/GAMMA)
        return white_response(self.fl)*t*(1-t)*self.unit(t,hue)

    def boundary(self,h,checkpoint=None):return Boundary(self).intersection(h,checkpoint)

    def _map(self,q,energy,operation,checkpoint=None):
        if checkpoint is not None:checkpoint()
        q=np.asarray(q,dtype=float);e=np.broadcast_to(np.asarray(energy,dtype=float),q.shape[:-1])
        if q.shape[-1:]!=(3,) or not np.isfinite(q).all() or not np.isfinite(e).all():
            raise ValueError('Spindle requires finite triples and energy')
        out=q.copy();eo=np.array(e,copy=True)
        if operation=='guard':
            h=q[...,0];black=h<=0;tip=h>=self.white
            out[black]=0.;eo[black]=0.;out[tip]=[self.white,0.,0.];eo[tip]=self.white**2
            mask=~black&~tip
            if np.any(mask):
                v=q[mask];cap,eb=self.boundary(v[:,0],checkpoint)
                change=e[mask]<eb
                c=np.hypot(v[:,1],v[:,2]);f=np.divide(cap,c,out=np.ones_like(c),where=c>0)
                out[mask,1:]=v[:,1:]*np.where(change,f,1.)[:,None]
                eo[mask]=np.where(change,eb,e[mask])
                values=out[mask];energies=eo[mask]
                for i in np.flatnonzero(v[:,0]<1e-140):
                    a,b=self.guard_scaled(list(map(S,v[i])),S(e[mask][i]));values[i]=list(map(float,a));energies[i]=float(b)
                out[mask]=values;eo[mask]=energies
            if checkpoint is not None:checkpoint()
            return out,eo
        if np.any(e<0):raise ValueError('Finite mapping requires nonnegative energy')
        j=np.sqrt(e)
        if np.any(j>self.white):raise ValueError('J_A exceeds cube white')
        c=np.hypot(q[...,1],q[...,2]);mask=(j>0)&(j<self.white)&(c>0)
        t=(j[mask]/self.white)**(1/GAMMA);g=self.coefficient(t)
        hue=np.arctan2(q[mask,2],q[mask,1]);b=self.white*t*(1-t)*g
        k=white_response(self.fl)/self.white*self.unit(t,hue,checkpoint)/g
        if np.any(k<1-1e-12):raise ValueError('Round profile exceeds cube')
        dest=map_radius(c[mask],b,np.maximum(k,1.),operation=='release')
        out[mask,1:]=q[mask,1:]*(dest/c[mask])[:,None]
        out[...,0]=np.hypot(j,np.sqrt(66*np.hypot(out[...,1],out[...,2])))
        tips=(j==0)|(j==self.white);out[tips,1:]=0.;out[tips,0]=j[tips]
        return out,eo

    def guard(self,q,e,checkpoint=None):return self._map(q,e,'guard',checkpoint)
    def compress(self,q,e,checkpoint=None):return self._map(q,e,'compress',checkpoint)
    def release(self,q,e,checkpoint=None):return self._map(q,e,'release',checkpoint)

    def _body_scaled(self,t):return self.white*t*(1-t)*float(self.coefficient(float(t)))

    def guard_scaled(self,q,energy):
        h=q[0];c=norm(q[1:])
        if h<=0:return [S(),S(),S()],S()
        if h>=self.white:return [S(self.white),S(),S()],S(self.white)**2
        target,e=Boundary(self).intersection_scaled(h)
        if energy>=e:return q,energy
        factor=target/c if c else S(1)
        return [h,q[1]*factor,q[2]*factor],e

    def finite_scaled(self,q,energy,release=False):
        if energy<0:raise ValueError('Finite mapping requires nonnegative energy')
        j=energy**.5;c=norm(q[1:])
        if j>self.white:raise ValueError('J_A exceeds cube white')
        if not j or j==self.white or not c:return [j,S(),S()],energy
        t=(j/self.white)**(1/GAMMA);g=float(self.coefficient(float(t)));b=self._body_scaled(t)
        h=np.arctan2(float(q[2]/c),float(q[1]/c))
        k=float(white_response(self.fl)/self.white*self.unit(float(t),h)/g)
        if k<1-1e-12:raise ValueError('Round profile exceeds cube')
        dest=b*float(map_radius(float(min(c/b,S(1 if release else k))),1.,max(k,1.),release))
        return [norm([j,(66*dest)**.5]),q[1]*(dest/c),q[2]*(dest/c)],energy

    def responses(self,q,energy):
        q=np.asarray(q);energy=np.asarray(energy)
        if np.any(energy<0):raise ValueError('Decode requires nonnegative energy')
        hue=np.arctan2(q[...,2],q[...,1])
        opponents=q[...,1:]*(self.aw/(1204*eccentricity(hue)))[...,None]
        r=white_response(self.fl)*(np.sqrt(energy)/self.white)**(1/GAMMA)
        return r[...,None]+opponents@INVERSE[:,1:].T

    def cones(self,q,e):
        r=self.responses(q,e);rw=white_response(self.fl)
        if not np.isfinite(r).all() or np.any(abs(r)>=400):raise ValueError('Response pole reached')
        return np.sign(r)*MAXIMUM*(abs(r)*(400-rw)/(rw*(400-abs(r))))**(50/21)

    def decode(self,q,e):
        q=np.asarray(q);e=np.broadcast_to(e,q.shape[:-1]);result=image(self.cones(q,e),self.matrix)
        position=(np.sqrt(e)/self.white)**(1/GAMMA);mask=position>.5
        if np.any(mask):
            v=q[mask];hue=np.arctan2(v[:,2],v[:,1]);rw=white_response(self.fl)
            opponents=v[:,1:]*(self.aw/(1204*eccentricity(hue)))[:,None]
            d=(1-position[mask,None])-(opponents@INVERSE[:,1:].T)/rw
            valid=np.all(d<=1,axis=-1);d=d[valid]
            deficits=-MAXIMUM*np.expm1((50/21)*(np.log1p(-d)-np.log1p(d*rw/(400-rw))))
            selected=result[mask];selected[valid]=MAXIMUM-image(deficits,self.matrix);result[mask]=selected
        return result

    def decode_scaled(self,q,energy):
        if energy<0:raise ValueError('Decode requires nonnegative energy')
        c=norm(q[1:]);h=np.arctan2(float(q[2]/c),float(q[1]/c)) if c else 0.
        rw=white_response(self.fl);t=(energy**.5/self.white)**(1/GAMMA);n=rw*t
        ab=[x*float(self.aw/(1204*eccentricity(h))) for x in q[1:]]
        r=[n+dot(row,ab) for row in INVERSE[:,1:]]
        if any(abs(x)>=400 for x in r):raise ValueError('Response pole reached')
        if t>.5 and all(x>=0 for x in r):
            gaps=np.array([float((1-t)-dot(row,ab)/rw) for row in INVERSE[:,1:]])
            deficits=-MAXIMUM*np.expm1((50/21)*(np.log1p(-gaps)-np.log1p(gaps*rw/(400-rw))))
            return [S(MAXIMUM)-S(deficits[1])-dot(row,[S(v)-deficits[1] for v in deficits]) for row in self.matrix]
        cones=[(1 if x>=0 else -1)*MAXIMUM*(abs(x)*(400-rw)/(rw*(400-abs(x))))**(50/21) for x in r]
        return [cones[1]+dot(row,[x-cones[1] for x in cones]) for row in self.matrix]

    @classmethod
    def load(cls,path,fl,aw,matrix):
        with np.load(path,allow_pickle=False) as d:
            if int(d['format'])!=FORMAT:
                raise ValueError(f'Spindle cache format {int(d["format"])}; expected {FORMAT}')
            if 'profile_certified' not in d or not bool(d['profile_certified']):raise ValueError('Uncertified spindle profile')
            if sum(v.nbytes for v in d.values())>BUDGET or Path(path).stat().st_size>BUDGET:
                raise ValueError('Aggregate context exceeds 64 KiB')
            tolerance=64*np.finfo(np.float64).eps
            for name,value in (('fl',fl),('aw',aw)):
                cached=float(d[name])
                if not np.isfinite([value,cached]).all() or abs(value-cached)>tolerance*(1+abs(cached)):
                    raise ValueError(f'D65 cache metadata mismatch for {name}: {value!r}; cached {cached!r}')
            matrix=np.asarray(matrix,dtype=np.float64);adapters=d['matrices']
            if matrix.shape!=(3,3) or adapters.shape!=(2,3,3) or not np.isfinite(matrix).all() or not np.isfinite(adapters).all():
                raise ValueError('D65 cache matrix metadata requires finite 3 by 3 adapters')
            distance=np.max(abs(adapters-matrix)/(1+abs(adapters)),axis=(1,2))
            index=int(np.argmin(distance))
            if distance[index]>tolerance:
                raise ValueError(f'D65 cache matrix metadata mismatch: scaled error {distance[index]:.17g}; limit {tolerance:.17g}')
            # Matching identifies a certified adapter; the runtime uses its
            # stored constants, never the slightly different recomputation.
            return cls(d['knots'].copy(),d['profile'].copy(),float(d['white']),
                       float(d['fl']),float(d['aw']),adapters[index].copy(),MappingProxyType(load_data(d)))


@lru_cache(maxsize=2)
def _context(fl,aw,matrix):
    return Context.load(Path(__file__).with_name('d65_boundary.npz'),fl,aw,np.asarray(matrix).reshape(3,3))


def context(fl,aw,matrix):return _context(float(fl),float(aw),tuple(np.asarray(matrix).ravel()))
