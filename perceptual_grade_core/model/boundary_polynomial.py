"""Shared uint16 face tables and polynomial crease/profile/guard lookup.

The compact source is sampled once; runtime faces use affine uint16 bilinear
lookup. This adopted approximation preserves shared creases, not C2 face
smoothness or strict cube containment. The geometric oracle remains separate.
"""
import numpy as np
from .boundary_table import Table, basis
from .creases import Selector,topology

SAMPLES=np.array([0.,1/3,2/3,1.])
INVERSE=np.linalg.inv(np.vander(SAMPLES,4,increasing=True))
FIRST=np.array([0]+[(k&-k).bit_length()-1 for k in range(1,32)],dtype=np.int32)
FACE_SIZE=257
LOOKUP_KEY=b'face-u16-257-bilinear-v1'
TAU=np.float32(2*np.pi)


def wrap_hue(values):
    """Float32 remainder with exact bounded additions on [-2τ,2τ).

    Keep the original unreduced hue in distance calculations: normalizing it
    first changes rounding at crease ties. Unbounded finite hues use remainder.
    """
    values=np.asarray(values,dtype=np.float32)
    # The bounded expression wins for large vectors. On the baker's small
    # concurrent chunks, its many NumPy calls cost more than one C ufunc.
    if values.size<=16384:return np.remainder(values,TAU)
    offset=np.where(values< -TAU,2*TAU,np.where(values<0,TAU,np.where(values>=TAU,-TAU,np.float32(0))))
    result=np.asarray(values+offset)
    outside=(values< -2*TAU)|(values>=2*TAU)
    if np.any(outside):result[outside]=np.remainder(values[outside],TAU)
    return result


def pieces(knots):
    axis=np.unique(knots).astype(float)
    points=axis[:-1,None]+np.diff(axis)[:,None]*SAMPLES
    return axis,np.einsum('ij,pjk->pik',INVERSE,basis(knots,points.ravel()).reshape(-1,4,len(knots)-4))


def address(axis,values):
    i=np.clip(np.searchsorted(axis,values,side='right')-1,0,len(axis)-2)
    return i,(values-axis[i])/(axis[i+1]-axis[i])


def uniform_address(axis,values):
    """Arithmetic cell address, corrected against stored float32 knot ties."""
    i=np.clip((values*(len(axis)-1)).astype(np.int32),0,len(axis)-2)
    i=np.where(values<axis[i],np.maximum(0,i-1),i)
    i=np.where((values>=axis[i+1])&(i<len(axis)-2),i+1,i)
    return i,(values-axis[i])/(axis[i+1]-axis[i])


def edge_hints(expanded):
    """Conservative height/hue shortlists, never a sampled face-ID decision.

    Bound each cubic on each height cell by its restricted Bernstein control
    hull. An edge is omitted only if its smallest possible circular distance
    exceeds another edge's largest. Float32 padding includes cell-address and
    hue-wrap rounding; vertex ties and large unreduced hues use the full ring.
    The runtime still compares actual crease polynomials for all candidates.
    """
    heights=expanded['heights'].astype(float);vertices=expanded['vertex_t'].astype(float)
    axis=expanded['edge_axis'].astype(float);coeff=expanded['edge_polynomial'].astype(float)
    edges=expanded['edges'];rings=expanded['topology'];period=float(np.float32(2*np.pi))
    bins=128;rows=64;h0=np.arange(bins)*period/bins;h1=h0+period/bins
    hints=np.empty((7,rows,bins),dtype=np.uint8);eps=np.finfo('f4').eps
    for band in range(7):
        cells=np.linspace(heights[band],heights[band+1],rows+1)
        candidates=rings[2*band+1,:,0];unique=np.r_[True,[e not in candidates[:k] for k,e in enumerate(candidates[1:],1)]]
        for row in range(rows):
            lower=[];upper=[]
            for edge in candidates:
                lo,hi=vertices[edges[edge]]
                x0,x1=np.clip((cells[row:row+2]+np.array([-16,16])*eps-lo)/(hi-lo),0,1)
                ids=np.flatnonzero((axis[:-1]<=x1)&(axis[1:]>=x0))
                c=coeff[edge,ids,:,0];width=axis[ids+1]-axis[ids]
                p=np.maximum(0,(x0-axis[ids])/width);q=np.minimum(1,(x1-axis[ids])/width);d=q-p
                a=((c[:,3]*p+c[:,2])*p+c[:,1])*p+c[:,0]
                b=(c[:,1]+2*c[:,2]*p+3*c[:,3]*p*p)*d
                cc=(c[:,2]+3*c[:,3]*p)*d*d;dd=c[:,3]*d*d*d
                control=np.stack((a,a+b/3,a+2*b/3+cc/3,a+b+cc+dd),-1)
                margin=32*eps*np.max(np.sum(abs(c),-1))+64*eps*period
                lower.append(control.min()-margin);upper.append(control.max()+margin)
            low=h0[:,None]-np.asarray(upper);high=h1[:,None]-np.asarray(lower)
            crosses=np.floor(low/period)!=np.floor(high/period)
            minimum=np.where(crosses,0,low%period);maximum=np.where(crosses,period,high%period)
            keep=(minimum<=maximum.min(-1,keepdims=True))&unique
            hints[band,row]=np.sum(keep*(1<<np.arange(5)),axis=-1).astype(np.uint8)
    return hints


def face_polynomials(data):
    """Temporary float32 cells used to sample the unchanged compact source."""
    radius_axis,radius_basis=pieces(data['sector_s'])
    height_axes=[];faces=[]
    for axis,coefficients in zip(data['sector_t'],data['sectors']):
        height_axis,height_basis=pieces(axis)
        height_axes.append(height_axis)
        faces.append(np.einsum('pia,ab,qjb->pqij',height_basis,coefficients.astype(float),radius_basis,optimize=True))
    return {name:np.asarray(value,dtype=np.float32) for name,value in dict(
        height_axes=height_axes,radius_axis=radius_axis,face_polynomial=faces).items()}


def sample_faces(data):
    """Match the adopted experiment's float32 samples and nearest uint16 codec."""
    source=face_polynomials(data);grid=np.linspace(0,1,FACE_SIZE,dtype=np.float32)
    x,s=np.meshgrid(grid,grid,indexing='ij');x=x.ravel();s=s.ravel()
    values=np.empty((6,FACE_SIZE*FACE_SIZE),dtype=np.float32)
    for face in range(6):
        for start in range(0,len(x),16384):
            stop=min(start+16384,len(x));ti,tx=address(source['height_axes'][face],x[start:stop])
            si,sx=address(source['radius_axis'],s[start:stop]);sx=sx[:,None]
            c=source['face_polynomial'][face,ti,si]
            v=((c[...,3]*sx+c[...,2])*sx+c[...,1])*sx+c[...,0]
            values[face,start:stop]=((v[:,3]*tx+v[:,2])*tx+v[:,1])*tx+v[:,0]
    offset=values.min(axis=1);scale=((values.max(axis=1).astype(float)-offset)/65535).astype(np.float32)
    if np.any(scale<=0):raise ValueError('Degenerate fixed-point face range')
    codes=np.rint((values.astype(float)-offset[:,None])/scale[:,None]).clip(0,65535).astype(np.uint16)
    low=data['vertex_t'][data['face_limits'][:,0]]
    span=data['vertex_t'][data['face_limits'][:,1]]-low
    result=dict(face_codes=codes.reshape(6,FACE_SIZE,FACE_SIZE),face_offset=offset,
                face_scale=scale,face_low=low,face_span=span)
    for value in result.values():value.flags.writeable=False
    return result


def expand(data):
    edge_axis,edge_basis=pieces(data['edge_knots'])
    edge=np.einsum('pja,eac->epjc',edge_basis,data['edge_coefficients'].astype(float))
    curves=dict(edge_axis=edge_axis,edge_polynomial=edge)
    if 'guard_knots' in data:
        guard_axis,guard_basis=pieces(data['guard_knots'])
        guard=np.einsum('pja,a->pj',guard_basis,data['guard_coefficients'].astype(float))
        curves.update(guard_axis=guard_axis,guard_polynomial=guard)
    if 'knots' in data:
        # Historical triangularly rounded profile is also piecewise cubic.
        knots=data['knots'].astype(float);radius=.25*np.minimum(np.diff(knots)[:-1],np.diff(knots)[1:])
        profile_axis=np.unique(np.r_[knots,knots[1:-1]-radius,knots[1:-1]+radius])
        x=profile_axis[:-1,None]+np.diff(profile_axis)[:,None]*SAMPLES
        values=data['profile'];value=np.interp(x,knots,values)
        slopes=np.diff(values)/np.diff(knots)
        for i,r in enumerate(radius,1):
            v=np.maximum(0,r-abs(x-knots[i]))
            value+=(slopes[i]-slopes[i-1])*v*v*v/(6*r*r)
        curves.update(profile_axis=profile_axis,profile_polynomial=np.einsum('ij,pj->pi',INVERSE,value))
    result={name:np.asarray(value,dtype=np.float32) for name,value in curves.items()}
    result.update(sample_faces(data))
    for name in ('vertex_t','edges','edge_faces','face_limits'):result[name]=np.asarray(data[name]).copy()
    result['heights'],result['topology']=topology(result['vertex_t'],result['edges'],result['edge_faces'],result['face_limits'])
    result['edge_hints']=edge_hints(result)
    for value in result.values():value.flags.writeable=False
    return result


class PolynomialSelector(Selector):
    def __init__(self,data,expanded):
        super().__init__(data,np.float32)
        self.expanded=expanded

    def hint_mask(self,t,hue,region):
        band=np.minimum(region//2,6);table=self.expanded['edge_hints']
        x=(t-self.heights[band])/(self.heights[band+1]-self.heights[band])
        i=np.clip((x*table.shape[1]).astype(np.int32),0,table.shape[1]-1)
        j=np.minimum((wrap_hue(hue)/(2*np.pi)*table.shape[2]).astype(np.int32),table.shape[2]-1)
        return np.where((region%2==0)|(abs(hue)>8*np.pi),31,table[band,i,j])

    def left(self,t,hue,region,retain=False):
        mask=self.hint_mask(t,hue,region);slot=FIRST[mask].copy()
        edge=self.topology[region,slot,0];i,z=self.edge_address(t,edge)
        angle=wrap_hue(self.edge_value(edge,i,z,0))
        distance=wrap_hue(hue-angle);remaining=mask&~(1<<slot)
        active=np.flatnonzero(remaining);remaining=remaining[active]
        for k in range(5):
            ids=active[(remaining&(1<<k))!=0]
            if not len(ids):continue
            edge=self.topology[region[ids],k,0];oi,oz=self.edge_address(t[ids],edge)
            other=wrap_hue(self.edge_value(edge,oi,oz,0))
            d=wrap_hue(hue[ids]-other)
            take=(d<distance[ids])|((d==distance[ids])&(k<slot[ids]));chosen=ids[take]
            slot[chosen]=k;distance[chosen]=d[take];angle[chosen]=other[take]
            if retain:i[chosen]=oi[take];z[chosen]=oz[take]
        result=(self.topology[region,slot],distance,angle)
        return (*result,i,z) if retain else result

    def edge_address(self,t,edge):
        low=self.vertices[self.edges[edge,0]];high=self.vertices[self.edges[edge,1]]
        x=np.clip((t-low)/(high-low),0,1)
        return uniform_address(self.expanded['edge_axis'],x)

    def edge_value(self,edge,i,z,channel):
        c=self.expanded['edge_polynomial'][edge,i,:,channel]
        return ((c[...,3]*z+c[...,2])*z+c[...,1])*z+c[...,0]

    def curve(self,t,edge,channel):
        i,z=self.edge_address(t,edge)
        return self.edge_value(edge,i,z,channel)

    def select(self,t,hue):
        """Keep each sample's selected hue addresses for its radius lookups."""
        t,hue=np.broadcast_arrays(np.asarray(t,dtype=np.float32),np.asarray(hue,dtype=np.float32))
        shape=t.shape;t,hue=t.ravel(),hue.ravel()
        chosen,dl,left_angle,li,lz=self.left(t,hue,self.region(t),retain=True)
        left,r0,r1,face=chosen.T;right=r0.copy()
        ri,rz=self.edge_address(t,right)
        right_angle=wrap_hue(self.edge_value(right,ri,rz,0))
        dr=wrap_hue(right_angle-hue)
        ties=np.flatnonzero(r0!=r1)
        if len(ties):
            oi,oz=self.edge_address(t[ties],r1[ties])
            angle=wrap_hue(self.edge_value(r1[ties],oi,oz,0))
            distance=wrap_hue(angle-hue[ties]);take=distance<dr[ties];ids=ties[take]
            right[ids]=r1[ids];right_angle[ids]=angle[take];dr[ids]=distance[take]
            ri[ids]=oi[take];rz[ids]=oz[take]
        fraction=np.divide(dl,dl+dr,out=np.zeros_like(dl),where=dl+dr>0)
        values=(face,fraction,left_angle,right_angle,
                self.edge_value(left,li,lz,1),self.edge_value(right,ri,rz,1))
        return tuple(value.reshape(shape) for value in values)

    def curves(self,t):
        t=np.asarray(t,dtype=np.float32)
        low,high=self.edges.T;x=(t[...,None]-self.vertices[low])/(self.vertices[high]-self.vertices[low])
        active=(x>=0)&(x<=1)
        values=np.zeros((*x.shape,2),dtype=np.float32)
        for edge in range(12):
            ids=active[...,edge]
            if not np.any(ids):continue
            i,z=uniform_address(self.expanded['edge_axis'],x[...,edge][ids]);z=z[...,None]
            c=self.expanded['edge_polynomial'][edge,i]
            values[...,edge,:][ids]=((c[...,3,:]*z+c[...,2,:])*z+c[...,1,:])*z+c[...,0,:]
        return values,active


class PolynomialTable(Table):
    def __init__(self,data,expanded=None):
        super().__init__(data,np.float32,expand(data) if expanded is None else expanded)
        self.selector=PolynomialSelector(data,self.expanded)

    def height_factor(self,height):
        return self._curve('guard',height)

    def coefficient(self,t):
        return self._curve('profile',t)

    def _curve(self,name,x):
        i,z=address(self.expanded[name+'_axis'],np.asarray(x,dtype=np.float32))
        c=self.expanded[name+'_polynomial'][i]
        return ((c[...,3]*z+c[...,2])*z+c[...,1])*z+c[...,0]
