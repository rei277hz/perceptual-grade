"""Direct fitted boundary for float64/extended-range appearance operations.

The independent iterative geometric oracle lives in tools/build and is never
installed. Float64 and CPU/GPU float32 paths share the adopted uint16 face
samples; edge, profile and guard evaluation retain their cubic curves.
"""
import numpy as np
from ...color_core.modcam16 import GAMMA
from ...color_core.scaled import Scalar as S


class Boundary:
    def __init__(self,ctx):
        self.ctx=ctx;self.table=ctx.lookup

    def capacity(self,t,h,checkpoint=None):
        if checkpoint is not None:checkpoint()
        result=self.table.capacity(t,h)
        if checkpoint is not None:checkpoint()
        return result

    def intersection(self,h,checkpoint=None):
        if checkpoint is not None:checkpoint()
        h=np.asarray(h,dtype=float)
        if not np.isfinite(h).all() or np.any((h<=0)|(h>self.ctx.white)):
            raise ValueError('Intersection requires positive height up to cube white')
        c,e=self.table.intersection(h,self.ctx.white,GAMMA)
        tiny=h<1e-140
        if np.any(tiny):
            c,e=np.asarray(c),np.asarray(e)
            for i in np.flatnonzero(tiny.ravel()):
                a,b=self.intersection_scaled(S(h.ravel()[i]))
                c.ravel()[i],e.ravel()[i]=float(a),float(b)
        if checkpoint is not None:checkpoint()
        return c,e

    def intersection_scaled(self,h):
        if h<=0 or h>self.ctx.white:raise ValueError('Intersection requires positive height up to cube white')
        x=h/self.ctx.white
        t=x*x*float(self.table.height_factor(float(x)))
        j=min(h,self.ctx.white*min(S(1),t)**GAMMA)
        return (h-j)*(h+j)/66,j*j
