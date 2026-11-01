"""Portable Stage 3 scope preparation and CPU diagnostics.

The host adapter owns GPU scheduling and publication. These functions retain
only value arrays, immutable model records, and NumPy calculations.
"""

from __future__ import annotations

from collections import OrderedDict
import numpy as np

from . import color, tone, scope_analysis, scope_color, statistics as robust, scope_grid
from ...color_core import modcam16 as responses
from . import scope_cloud
from .forward import Model
from .saturation import project, bounded_project

SCHEMA = 2

def _check(checkpoint):
    if checkpoint is not None:checkpoint()


def demands(snapshot):
    values=snapshot.get('_scope_demands',(snapshot.get('view',(0,True)),))
    return tuple(sorted({(int(t),bool(w)) for t,w in values if 0<=int(t)<4}))


def _rgba(value,shape=None):
    """Skip invalid raw samples before any chain or appearance calculation."""
    rgba=np.asarray(value,dtype=float)
    if rgba.ndim!=3 or rgba.shape[-1]!=4 or min(rgba.shape[:2])<=0:
        raise ValueError('Scope source must be a nonempty H x W x 4 RGBA array')
    if shape is not None and rgba.shape!=shape:
        raise ValueError('Scope reference RGBA shape must match the primary sample grid')
    valid=np.isfinite(rgba).all(axis=-1)&(rgba[...,3]>0)
    safe=np.where(valid[...,None],rgba,0.)
    safe[...,3]=np.clip(safe[...,3],0,1)
    return safe,valid


def prepare(snapshot,cache,checkpoint=None):
    """Register source/Stage 2 once; cache ownership belongs to the helper."""
    _check(checkpoint)
    s=snapshot
    key=(SCHEMA,scope_color.MEMBERSHIP_SCHEMA,repr(s['base_key']),
         repr(s.get('_scope_source_cache_id',s.get('_scope_cache_id',s.get('cache_id')))),
         repr(s.get('_scope_provenance',())),repr(scope_grid.key(s.get('sample_grid'))))
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    rgba,source_valid=_rgba(s['rgba'])
    mass,x=scope_grid.values(s.get('sample_grid'),source_valid.shape)
    source_valid=source_valid&(mass>0)
    model=tone.model(s['tone'],s['grading_space'])
    if s['stack']:
        incoming=scope_analysis.apply_chain(rgba[...,:3],s['chain'],s['space'])
        rgb=scope_analysis.evaluate_signal(incoming,s['signal'])@s['to_grade'].T
        appearance=Model(model).encode_cones(model.cones(rgb))
        base=dict(model=model,alpha=rgba[...,3],stage2_appearance=appearance,
                  stage2_input_rgb=rgb,stack_output_rgb=rgb)
    else:
        base=scope_analysis.evaluate_samples(rgba,s['linear'],s['tone'],s['space'],
            s['chain'],model,s['matrix'],s['to_grade'],s.get('sample_grid'))
    _check(checkpoint)
    appearance=base['stage2_appearance'];q=appearance.q
    alpha=np.where(np.isfinite(base['alpha']),np.clip(base['alpha'],0,1),0.)
    valid=source_valid&np.isfinite(q).all(axis=-1)&np.isfinite(appearance.ja)&(alpha>0)
    H=np.hypot(appearance.ja,np.sqrt(66*np.hypot(q[...,1],q[...,2])))
    cm=color.model(s['tone'],s['params'],s['grading_space'])
    groups=scope_color.tone_groups(cm,H,valid)
    base.update(tone_id=groups,alpha=alpha,H=H,groups={},outputs=OrderedDict(),
                label=s.get('label','Input'),registration_key=key,comparison=None,
                sample_mass=mass,sample_x=x,sample_grid=s.get('sample_grid'))
    if s.get('comparison') is not None:
        rgba,chain=s['comparison']
        rgba,_=_rgba(rgba,source_valid.shape+(4,))
        rgb=scope_analysis.apply_chain(rgba[...,:3],chain,s['space'])@s['to_grade'].T
        ref=Model(model).encode_cones(model.cones(rgb))
        refalpha=np.where(np.isfinite(rgb).all(axis=-1)&np.isfinite(rgba[...,3]),
                          np.clip(rgba[...,3],0,1),0.)
        base['comparison']=dict(q=ref.q,energy=ref.ja*abs(ref.ja),rgb=rgb,alpha=refalpha)
    for field in ('before','after','before_j','after_j','source_rgb'):
        base.pop(field,None)
    cache[key]=base
    while len(cache)>4:cache.popitem(last=False)
    return base


def group(base,model,index,checkpoint=None):
    """Compact one stable source group, retaining original image x and UV IDs."""
    if index in base['groups']:return base['groups'][index]
    _check(checkpoint)
    ids=np.flatnonzero(base['tone_id'].reshape(-1)==index)
    app=base['stage2_appearance'];q=app.q.reshape(-1,3)[ids]
    ja=app.ja.reshape(-1)[ids];alpha=base['alpha'].reshape(-1)[ids]
    mass=base['sample_mass'].reshape(-1)[ids]
    energy=ja*abs(ja)
    near=scope_color.near_neutral(q,energy)
    _,sample_ids=scope_grid.sample(q[:,1:],alpha*near,mass,keys=ids)
    width=1
    result=dict(ids=ids,q=q,energy=energy,alpha=alpha,capacity=None,
                mass=mass,x=base['sample_x'].reshape(-1)[ids].astype(np.float32),width=width,sample_ids=sample_ids,
                rgb=base['stage2_input_rgb'].reshape(-1,3)[ids],
                H=base['H'].reshape(-1)[ids],histograms=None,gpu=None)
    if base['comparison'] is not None:
        ref=base['comparison']
        result['comparison']=dict(q=ref['q'].reshape(-1,3)[ids],
            energy=ref['energy'].reshape(-1)[ids],rgb=ref['rgb'].reshape(-1,3)[ids],
            alpha=alpha*ref['alpha'].reshape(-1)[ids],mass=mass,x=result['x'],width=width)
    else:result['comparison']=None
    base['groups'][index]=result
    return result


def _rec709(rgb,space):
    from . import linear
    matrix=(np.eye(3) if isinstance(space,linear.GradeSpace) else
            np.asarray(linear.xyz_to_rgb('Linear Rec.709'))@np.asarray(linear.rgb_to_xyz(space)))
    return np.asarray(rgb)@matrix.T


def _cpu_output(base,model,selected,checkpoint=None):
    _check(checkpoint)
    if 'stack_output_rgb' in base or not(model.tone.active or model.active):
        q,energy,rgb=selected['q'],selected['energy'],selected['rgb']
    else:
        engine=Model(model.tone,model)
        guarded=engine.guarded_appearance(responses.State(selected['q'],selected['energy']),checkpoint)
        if guarded is None:
            # The authoritative scaled implementation remains available for
            # controls outside the GPU/vector arithmetic domain.
            rgb=engine.apply(base['stage1_rgb'].reshape(-1,3)[selected['ids']],checkpoint)
            app=engine.encode_cones(model.tone.cones(rgb));q=app.q;energy=app.ja*abs(app.ja)
        else:
            q,energy,native=guarded;rgb=engine.from_native(native)
    result=dict(q=q,energy=energy,rgb=rgb,alpha=selected['alpha'],mass=selected['mass'],x=selected['x'],width=selected['width'],backend='cpu')
    result['comparison']=selected['comparison']
    return result



def density_levels(values):
    values=np.asarray(values)
    maximum=max(1.,float(np.max(values,initial=0.)))
    return np.clip(np.rint(np.log1p(np.maximum(values,0))/np.log1p(maximum)*7),0,7).astype(np.uint8)


def _cpu_jc(signal,model):
    q=signal['q'];alpha=signal['alpha']*signal.get('mass',1.);H=q[:,0];Hmax=model.tone.plot_max*100
    valid=np.isfinite(H)&(alpha>0);inside=valid&(H>=0)&(H<=Hmax)
    wave=np.zeros((160,192),np.float64)
    yi=np.minimum(159,(H[inside]*160/Hmax).astype(int))
    xi=np.minimum(191,(signal['x'][inside].astype(np.float64)*192/signal['width']).astype(int))
    np.add.at(wave,(yi,xi),alpha[inside])
    vectors=scope_color.vectorscope(signal['rgb'],alpha,model.tone.space)
    return dict(waveform_levels=density_levels(wave),vectorscope_levels=density_levels(vectors['density']),
                waveform_outside=float(np.sum(alpha[valid&~inside])),vectorscope_outside=vectors['outside'])


def _histograms(selected,model):
    q=selected['q'];ja=np.sign(selected['energy'])*np.sqrt(abs(selected['energy']))
    if selected['capacity'] is None:selected['capacity']=model.capacity(q,ja)
    drivers=color.drivers(q,ja,selected['capacity'])
    bins=np.linspace(0,model.tone.plot_max*100,97)
    weights=selected['alpha']*selected.get('mass',1.)
    total=max(float(np.sum(weights)),1e-20)
    result={name:np.asarray(np.histogram(selected['H'],bins,
        weights=weights*drivers[:,i])[0]/total,dtype=np.float32)
        for i,name in enumerate(('low','high'))}
    result['sectors']=np.asarray(np.histogram(np.arctan2(q[:,2],q[:,1]),
        bins=np.linspace(-np.pi,np.pi,49),
        weights=weights*np.hypot(drivers[:,2],drivers[:,3]))[0]/total,dtype=np.float32)
    return result


def _views(snapshot,index):
    for raw in snapshot.get('_warp_views',()):
        if 'tone' in raw and int(raw['tone'])!=index:continue
        yield scope_cloud._descriptor(raw)


def _cpu_warp(output, view, index, checkpoint=None):
    """Produce the demanded compact display tile from retained CPU arrays."""
    layers = [output]
    if output['comparison'] is not None:
        layers.append(output['comparison'])
    tiles = [np.rint(np.clip(scope_cloud._raster_layer(
        bounded_project(value['q'], value['energy']), value['alpha'],
        value['energy'], view, index, checkpoint, mass=value.get('mass')), 0, 1) * 255).astype(np.uint8)
        for value in layers]
    return dict(view=view, tone=index, pixels=np.stack(tiles)[:, None],
                backend='cpu', timings={})


