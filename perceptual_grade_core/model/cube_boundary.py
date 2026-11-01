"""Independent Blender raw appearance decoder and fitted physical cube rays."""
from functools import lru_cache, cached_property
from pathlib import Path
from types import MappingProxyType
import numpy as np
from ...color_core.modcam16 import GAMMA, INVERSE, eccentricity
from ...color_core.scaled import Scalar as S, norm, dot
from .boundary_resource import FORMAT, BUDGET, load_data
MAXIMUM=1024.


def white_response(fl):
    p=np.power(np.float64(fl)*MAXIMUM,.42)
    return float(400*p/(27.13+p))


def image(values,matrix):
    return np.stack([values[...,1]+row[0]*(values[...,0]-values[...,1])+row[2]*(values[...,2]-values[...,1]) for row in matrix],axis=-1)


def opponent_responses(ab):
    """Fixed addition order for both isolated and batched validation."""
    return ab[...,0,None]*INVERSE[:,1]+ab[...,1,None]*INVERSE[:,2]


class Context:
    def __init__(self,white,fl,aw,matrix,cache):
        self.white,self.fl,self.aw,self.matrix,self.cache=white,fl,aw,matrix,cache
        matrix.setflags(write=False)

    @cached_property
    def lookup(self):
        from .boundary_table import Table
        return Table(self.cache)

    def unit(self,t,hue,checkpoint=None):
        if checkpoint is not None:checkpoint()
        return self.lookup.capacity(t,hue)

    def capacity(self,ja,hue):
        ja,hue=np.broadcast_arrays(np.asarray(ja,dtype=float),np.asarray(hue,dtype=float))
        valid=(ja>0)&(ja<self.white)
        t=np.clip(np.maximum(ja,0)/self.white,0,1)**(1/GAMMA)
        return np.where(valid,white_response(self.fl)*t*(1-t)*self.unit(t,hue),0.)

    def guard(self,q,e,checkpoint=None):
        from .cube_guard import guard
        if checkpoint is not None:checkpoint()
        return guard(self,q,e)

    def guard_scaled(self,q,e):
        from .cube_guard import guard_scaled
        return guard_scaled(self,q,e)

    def responses(self,q,energy):
        q=np.asarray(q);energy=np.asarray(energy)
        hue=np.arctan2(q[...,2],q[...,1])
        opponents=q[...,1:]*(self.aw/(1204*eccentricity(hue)))[...,None]
        r=white_response(self.fl)*np.sign(energy)*(np.sqrt(abs(energy))/self.white)**(1/GAMMA)
        return r[...,None]+opponent_responses(opponents)

    def cones(self,q,e):
        r=self.responses(q,e);rw=white_response(self.fl)
        if not np.isfinite(r).all() or np.any(abs(r)>=400):raise ValueError('Response pole reached')
        return np.sign(r)*MAXIMUM*(abs(r)*(400-rw)/(rw*(400-abs(r))))**(50/21)

    def decode(self,q,e):
        q=np.asarray(q);e=np.broadcast_to(e,q.shape[:-1]);result=image(self.cones(q,e),self.matrix)
        position=np.sign(e)*(np.sqrt(abs(e))/self.white)**(1/GAMMA);mask=position>.5
        if np.any(mask):
            v=q[mask];hue=np.arctan2(v[:,2],v[:,1]);rw=white_response(self.fl)
            opponents=v[:,1:]*(self.aw/(1204*eccentricity(hue)))[:,None]
            d=(1-position[mask,None])-opponent_responses(opponents)/rw
            valid=np.all(d<=1,axis=-1);d=d[valid]
            deficits=-MAXIMUM*np.expm1((50/21)*(np.log1p(-d)-np.log1p(d*rw/(400-rw))))
            selected=result[mask];selected[valid]=MAXIMUM-image(deficits,self.matrix);result[mask]=selected
        return result

    def decode_scaled(self,q,energy):
        c=norm(q[1:]);h=np.arctan2(float(q[2]/c),float(q[1]/c)) if c else 0.
        rw=white_response(self.fl);t=(1 if energy>=0 else -1)*(abs(energy)**.5/self.white)**(1/GAMMA);n=rw*t
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
                raise ValueError(f'Cube cache format {int(d["format"])}; expected {FORMAT}')
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
            return cls(float(d['white']),float(d['fl']),float(d['aw']),
                       adapters[index].copy(),MappingProxyType(load_data(d)))


@lru_cache(maxsize=2)
def _context(fl,aw,matrix):
    return Context.load(Path(__file__).with_name('d65_boundary.npz'),fl,aw,np.asarray(matrix).reshape(3,3))


def context(fl,aw,matrix):return _context(float(fl),float(aw),tuple(np.asarray(matrix).ravel()))


@lru_cache(maxsize=1)
def default_context():
    path=Path(__file__).with_name('d65_boundary.npz')
    with np.load(path,allow_pickle=False) as data:
        return Context.load(path,float(data['fl']),float(data['aw']),data['matrices'][0])
