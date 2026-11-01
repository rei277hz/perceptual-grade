"""Validation for the compact, crease-selected D65 resource."""
import numpy as np

FORMAT=16
LEGACY_FORMAT=15
BUDGET=65536
FIELDS=('edge_knots','edge_coefficients','vertex_t','edges','edge_faces','face_limits',
        'sector_t','sector_s','sectors','white','fl','aw','matrices','format','certified')
LEGACY_FIELDS=(*FIELDS,'guard_knots','guard_coefficients','knots','profile','profile_certified')


def spline_axis(value,name):
    a=np.asarray(value)
    if (a.dtype!=np.dtype('f4') or a.ndim!=1 or len(a)<8 or not np.isfinite(a).all()
            or np.any(a[:4]!=0) or np.any(a[-4:]!=1) or np.any(np.diff(a[3:-3])<=0)):
        raise ValueError('Invalid '+name+' spline axis')


def load_data(data):
    version=int(data['format'])
    if version not in (FORMAT,LEGACY_FORMAT):raise ValueError('Invalid boundary cache format')
    historical=version==LEGACY_FORMAT
    fields=LEGACY_FIELDS if historical else FIELDS
    if set(data)!=set(fields):raise ValueError('Incomplete or unexpected boundary resource fields')
    if sum(v.nbytes for v in data.values())>BUDGET:raise ValueError('Boundary resource exceeds 64 KiB')
    if historical:
        if not bool(data['profile_certified']):raise ValueError('Uncertified spindle profile')
        knots,profile=data['knots'],data['profile']
        if (knots.dtype!=np.dtype('f8') or knots.ndim!=1 or len(knots)<3 or profile.dtype!=np.dtype('f8')
                or profile.shape!=knots.shape or not np.isfinite(knots).all() or not np.isfinite(profile).all()
                or knots[0]!=0 or knots[-1]!=1 or np.any(np.diff(knots)<=0) or np.any(profile<=0)):
            raise ValueError('Invalid spindle profile')
    for name in ('white','fl','aw'):
        if np.shape(data[name]) or not np.isfinite(data[name]) or data[name]<=0:raise ValueError('Invalid '+name)
    matrices=data['matrices']
    if matrices.shape!=(2,3,3) or matrices.dtype!=np.dtype('f8') or not np.isfinite(matrices).all():
        raise ValueError('Invalid matrix adapters')
    for name in ('edge_knots','sector_s'):spline_axis(data[name],name)
    if historical:spline_axis(data['guard_knots'],'guard_knots')
    if data['sector_t'].ndim!=2 or data['sector_t'].shape[0]!=6:raise ValueError('Invalid face height axes')
    for a in data['sector_t']:spline_axis(a,'face height')
    shapes={'edge_coefficients':(12,len(data['edge_knots'])-4,2),
            'sectors':(6,data['sector_t'].shape[1]-4,len(data['sector_s'])-4),
            'vertex_t':(8,)}
    if historical:shapes['guard_coefficients']=(len(data['guard_knots'])-4,)
    for name,shape in shapes.items():
        a=data[name]
        if a.dtype!=np.dtype('f4') or a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid '+name)
    if np.any(data['edge_coefficients'][...,1]<=0) or (historical and np.any(data['guard_coefficients']<=0)):
        raise ValueError('Invalid positive boundary radii')
    vertices=data['vertex_t']
    if vertices[0]!=0 or vertices[-1]!=1 or np.any((vertices[1:-1]<=0)|(vertices[1:-1]>=1)):
        raise ValueError('Invalid vertex heights')
    for name,shape,maximum in (('edges',(12,2),7),('edge_faces',(12,2),5),('face_limits',(6,2),7)):
        a=data[name]
        if a.dtype!=np.dtype('u1') or a.shape!=shape or np.any(a>maximum):raise ValueError('Invalid '+name)
    edges=data['edges'];faces=data['edge_faces'];limits=data['face_limits']
    expected={(i,i|bit) for bit in (1,2,4) for i in range(8) if not i&bit}
    if set(map(tuple,edges.tolist()))!=expected or np.any(vertices[edges[:,0]]>=vertices[edges[:,1]]):
        raise ValueError('Invalid cube edge topology')
    for (lo,hi),adjacent in zip(edges,faces):
        axis=int(lo)^int(hi)
        actual={c+(3 if int(lo)&bit else 0) for c,bit in enumerate((4,2,1)) if bit!=axis}
        if set(map(int,adjacent))!=actual:raise ValueError('Invalid adjacent edge faces')
    for face,(lo,hi) in enumerate(limits):
        ids=np.array([i for i in range(8) if bool(i&(1<<(2-face%3)))==(face>=3)])
        if int(lo)!=ids[np.argmin(vertices[ids])] or int(hi)!=ids[np.argmax(vertices[ids])]:
            raise ValueError('Invalid face limits')
    result={name:np.asarray(data[name]).copy() for name in fields}
    for value in result.values():value.flags.writeable=False
    return result
