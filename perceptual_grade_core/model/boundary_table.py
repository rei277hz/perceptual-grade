"""Affine uint16 boundary lookup: shared crease traces plus one selected face.

Each face stores the curvature remaining after its two edge values are
interpolated. Its correction vanishes on either edge, so neighboring faces
share values without smoothing away their derivative discontinuity.
"""
import numpy as np


def stencil(knots, values):
    """Four nonzero clamped cubic B-splines (Cox-de Boor, fixed degree)."""
    x=np.asarray(values);k=np.asarray(knots);degree=3
    span=np.clip(np.searchsorted(k,x,side='right')-1,degree,len(k)-degree-2)
    weights=np.zeros((*x.shape,degree+1),dtype=x.dtype);weights[...,0]=1
    left=np.zeros_like(weights);right=np.zeros_like(weights)
    for order in range(1,degree+1):
        left[...,order]=x-k[span+1-order];right[...,order]=k[span+order]-x
        saved=np.zeros_like(x)
        for r in range(order):
            term=weights[...,r]/(right[...,r+1]+left[...,order-r])
            weights[...,r]=saved+right[...,r+1]*term
            saved=left[...,order-r]*term
        weights[...,order]=saved
    return span[...,None]+np.arange(-degree,1),weights


def basis(knots,values):
    ids,w=stencil(knots,np.asarray(values,dtype=float));out=np.zeros((len(values),len(knots)-4))
    np.put_along_axis(out,ids,w,axis=1)
    return out


class Table:
    def __init__(self,data,dtype=np.float64,expanded=None):
        from .creases import Selector
        from .boundary_polynomial import sample_faces
        self.dtype=np.dtype(dtype).type
        self.selector=Selector(data,dtype)
        self.expanded=sample_faces(data) if expanded is None else expanded
        self.guard_knots=np.asarray(data['guard_knots'],dtype=dtype) if 'guard_knots' in data else None
        self.guard_coefficients=np.asarray(data['guard_coefficients'],dtype=dtype) if 'guard_coefficients' in data else None

    def capacity(self,t,hue):
        t,hue=np.broadcast_arrays(np.asarray(t,dtype=self.dtype),np.asarray(hue,dtype=self.dtype))
        shape=t.shape;t=t.ravel();hue=hue.ravel()
        if not np.isfinite(t).all() or not np.isfinite(hue).all() or np.any((t<0)|(t>1)):
            raise ValueError('Capacity requires finite hue and neutral response in [0,1]')
        face,s,_,_,left,right=self.selector.select(t,hue)
        return self.field(t,face,s,left,right).reshape(shape)

    def field(self,t,face,s,left,right):
        """Four exact integer fetches, bilinear codes, then one affine decode."""
        d=self.expanded;n=d['face_codes'].shape[1]
        u=(t-d['face_low'][face])/d['face_span'][face]*(n-1);v=s*(n-1)
        i=np.clip(u.astype(np.int32),0,n-2);j=np.clip(v.astype(np.int32),0,n-2)
        a=u-i.astype(self.dtype);b=v-j.astype(self.dtype);index=(face*n+i)*n+j
        codes=d['face_codes'].ravel()
        q00=codes[index].astype(self.dtype);q01=codes[index+1].astype(self.dtype)
        row0=q00+a*(codes[index+n].astype(self.dtype)-q00)
        row1=q01+a*(codes[index+n+1].astype(self.dtype)-q01)
        correction=(row0+b*(row1-row0))*d['face_scale'][face]+d['face_offset'][face]
        return 1/((1-s)*left+s*right+s*(1-s)*correction)

    def height_factor(self,height):
        i,w=stencil(self.guard_knots,np.asarray(height,dtype=self.dtype))
        return np.sum(self.guard_coefficients[i]*w,axis=-1)

    def intersection(self,height,white,gamma,return_response=False):
        h=np.asarray(height,dtype=self.dtype);x=h/white
        t=x*x*self.height_factor(x)
        ja=np.minimum(h,white*np.minimum(t,1)**gamma)
        result=((h-ja)*(h+ja)/66,ja*ja)
        return (*result,np.minimum(t,1)) if return_response else result
