"""Numerical contribution, distribution and vectorscope analysis."""
from dataclasses import fields, is_dataclass, replace
import numpy as np
from ...color_core.modcam16 import encode_cones
from .saturation import project, bounded_project
from . import color as ch, statistics as robust, linear as cm, scope_grid
MEMBERSHIP_SCHEMA=4
INVALID_TONE=np.uint8(255)


def _immutable(value, memo=None):
    """Copy numerical cache data while preserving aliases and freezing arrays."""
    memo={} if memo is None else memo
    identity=id(value)
    if identity in memo:return memo[identity]
    if isinstance(value,np.ndarray):
        result=np.array(value,copy=True)
        result.setflags(write=False);memo[identity]=result;return result
    from .lut import Cube
    if isinstance(value,Cube):
        from copy import copy
        result=copy(value);memo[identity]=result
        for name,item in vars(value).items():setattr(result,name,_immutable(item,memo))
        return result
    if isinstance(value,dict):
        result={};memo[identity]=result
        result.update((key,_immutable(item,memo)) for key,item in value.items())
        return result
    if isinstance(value,tuple):
        result=tuple(_immutable(item,memo) for item in value);memo[identity]=result;return result
    if is_dataclass(value):
        result=replace(value,**{field.name:_immutable(getattr(value,field.name),memo)
                                for field in fields(value)})
        memo[identity]=result;return result
    return value


def project_rec709(rgb,space):
    matrix=np.eye(3) if isinstance(space,cm.GradeSpace) else np.asarray(cm.xyz_to_rgb('Linear Rec.709'))@np.asarray(cm.rgb_to_xyz(space))
    linear=np.asarray(rgb)@matrix.T;a=np.abs(linear)
    code=np.sign(linear)*np.where(a<.018,4.5*a,1.099*a**.45-.099)
    y=code@np.array([.2126,.7152,.0722])
    return np.stack([(code[...,2]-y)/1.8556,(code[...,0]-y)/1.5748],axis=-1)


def tone_groups(model,H,valid=None):
    """Exclusive input-tone partition; exact midpoint ties enter the upper tone.

    H is the unsigned Stage 2 tonal magnitude, evaluated in float64. This
    display-only partition never changes the smooth grading coefficients.
    """
    values=np.asarray(H,dtype=np.float64)
    anchors=np.asarray(model.anchors,dtype=np.float64)
    boundaries=anchors[:-1]+(anchors[1:]-anchors[:-1])*.5
    good=np.isfinite(values)
    if valid is not None:good&=np.asarray(valid,dtype=bool)
    return np.where(good,np.searchsorted(boundaries,values,side='right'),INVALID_TONE).astype(np.uint8)


def analysis_weights(model,H):
    """Compatibility view; caches retain one uint8 group per sample instead."""
    groups=tone_groups(model,H)
    return (groups[...,None]==np.arange(4)).astype(float)


def near_neutral(q,energy):
    """DetMCD eligibility in the linear appearance-saturation chart: s < .1.

    Use stored signed J_A squared, including the chart's neutral-black rule.
    This input-only filter never culls the active tone's displayed point cloud.
    """
    coordinates=project(q,energy)
    return (np.isfinite(q).all(axis=-1)&np.isfinite(coordinates).all(axis=-1)
            &(np.hypot(coordinates[...,0],coordinates[...,1])<.1))


def vectorscope(rgb,weights,space,size=144):
    """Rec.709 signal vectorscope preview; independent of scene working gamut.

    Signed Rec.709 transfer, Y' coefficients .2126/.7152/.0722. This is an
    explicitly labeled diagnostic, before any downstream display transform.
    Out-of-view samples are counted, never clamped into the plotted cloud.
    """
    coordinates=project_rec709(rgb,space);cb=coordinates[...,0];cr=coordinates[...,1]
    mask=np.isfinite(cb)&np.isfinite(cr)&(abs(cb)<=.65)&(abs(cr)<=.65)&(weights>0)
    hist=np.histogram2d(cr[mask],cb[mask],bins=size,range=[[-.65,.65],[-.65,.65]],weights=weights[mask])[0]
    outside=float(np.sum(weights[(~mask)&np.isfinite(cb)&np.isfinite(cr)]))
    return dict(density=hist,outside=outside)


def output_waveform(after,weights,Hmax,rows=160,cols=192,x=None):
    """Output H versus image x, weighted by the *input* tone membership.

    Negative, out-of-view and nonfinite results are excluded, never piled up
    at the display edge. Fractional membership/alpha stays fractional.
    """
    H=np.asarray(after)[...,0];weights=np.asarray(weights)
    xx=(np.broadcast_to(np.arange(H.shape[1])[None,:]/H.shape[1],H.shape)
        if x is None else np.asarray(x))
    valid=np.isfinite(H)&np.isfinite(weights)&(weights>0)
    inside=valid&(H>=0)&(H<=Hmax)
    yi=np.minimum(rows-1,(H[inside]*rows/Hmax).astype(int))
    xi=np.minimum(cols-1,(xx[inside]*cols).astype(int))
    wave=np.zeros((rows,cols));np.add.at(wave,(yi,xi),weights[inside])
    return dict(density=wave,outside=float(np.sum(weights[valid&~inside])))


def _input_analysis(base,model):
    """Build the Stage 2-only numerical analysis kept by the worker cache."""
    stack='stack_output_rgb' in base
    if stack:
        rgb=base['stack_output_rgb']
        from ..model.forward import Model
        signed=Model(model.tone,model).encode_cones(model.tone.cones(rgb))
    else:signed=base['stage2_appearance']
    before=signed.q
    JA=signed.ja
    mass,x=scope_grid.values(base.get('sample_grid'),base['alpha'].shape)
    good=np.isfinite(before).all(axis=-1)&(base['alpha']>0)&(mass>0)
    safe=np.where(good[...,None],before,0)
    H=np.hypot(JA,np.sqrt(66*np.hypot(safe[...,1],safe[...,2])));Hmax=model.tone.plot_max*100
    groups=tone_groups(model,H,good)
    alpha=np.where(good,np.clip(base['alpha'],0,1),0)
    JA=np.where(good,JA,0)
    capacity=model.capacity(safe,JA)
    drivers=ch.drivers(safe,JA,capacity)
    neutral=near_neutral(safe,JA*abs(JA))
    inside=good&(H>=0)&(H<=Hmax)
    yi=np.minimum(159,(H[inside]*160/Hmax).astype(int));xi=np.minimum(191,(x[inside]*192).astype(int))
    wave=np.zeros((160,192));np.add.at(wave,(yi,xi),mass[inside])
    result=dict(before=before,safe=safe,H=H,tone_id=groups,alpha=alpha,drivers=drivers,
        sample_mass=mass,sample_x=x,
        near_neutral=neutral,inside=inside,yi=yi,xi=xi,waveform=wave,
        bins=np.linspace(0,Hmax,97))
    return result


def _selection(data,tone):
    safe=data['safe'];H=data['H'];d=data['drivers']
    opacity=np.where(data['tone_id']==tone,data['alpha'],0.)
    weights=opacity*data['sample_mass'];total=max(float(np.sum(weights)),1e-20)
    bins=data['bins'];hist={}
    for i,(name,_,_) in enumerate(ch.DRIVERS):
        hist[name]=np.histogram(H,bins,weights=weights*d[...,i])[0]/total
    hist['tone']=np.histogram(H,bins,weights=weights)[0]/total
    h=np.arctan2(safe[...,2],safe[...,1]);magnitude=np.hypot(d[...,2],d[...,3])
    sectors=np.histogram(h,bins=np.linspace(-np.pi,np.pi,49),weights=weights*magnitude)[0]/total
    _,ids=scope_grid.sample(safe[...,1:].reshape(-1,2),
                           (opacity*data['near_neutral']).ravel(),data['sample_mass'])
    weighted=np.zeros_like(data['waveform'])
    np.add.at(weighted,(data['yi'],data['xi']),weights[data['inside']])
    result=dict(before=data['before'],weights=weights,opacity_weights=opacity,
        sample_mass=data['sample_mass'],sample_x=data['sample_x'],drivers=d,histograms=hist,
        bins=bins,sectors=sectors,sample_indices=ids,waveform=data['waveform'],waveform_selected=weighted)
    return result


def input_analysis_cache(base,model):
    """Return acyclic Stage 2 analysis and all deterministic selections."""
    data=_input_analysis(base,model)
    data['selections']=tuple(_selection(data,tone) for tone in range(4))
    # Keep the exclusive uint8 IDs and alpha once. Dense per-tone weights are
    # request-local compatibility views, not four retained membership arrays.
    for selected in data['selections']:
        selected.pop('weights',None);selected.pop('opacity_weights',None)
    return _immutable(data)


def input_analysis(base,model,data=None,prepare=None):
    """Materialize request-local links around cached numerical analysis."""
    from ..model.pipeline import Prepared
    if prepare is None:
        prepare='_input_cache' not in base
    data=_input_analysis(base,model) if data is None else data
    result=dict(data)
    result.pop('selections',None)
    result['_base']=base
    result['prepared']=(None if ('stack_output_rgb' in base or 'grade_output_rgb' in base or not prepare)
                        else Prepared(base,model))
    return result


def selection(base,model,tone,data=None,cached=None):
    data=input_analysis(base,model) if data is None else data
    result=dict(_selection(data,tone) if cached is None else cached)
    if 'weights' not in result:
        result['opacity_weights']=np.where(data['tone_id']==tone,data['alpha'],0.)
        result['weights']=result['opacity_weights']*data['sample_mass']
    result['_input']=data
    return result


def evaluate(base,params,space,tone,need_neutral=True,selected_model=None,_shared=None):
    space=base['model'].space
    model=selected_model or ch.model(base['model'].params,params,space)
    selected=selection(base,model,tone,None if _shared is None else _shared['input'],
                       None if _shared is None else _shared.get('selection'))
    if _shared is None and 'stack_output_rgb' not in base and 'grade_output_rgb' not in base:
        from ..model.pipeline import Prepared
        fallback_rgb=Prepared(base,model).apply(model)
    else:
        fallback_rgb=None
    after_rgb=(_shared['rgb'] if _shared is not None else
               base['stack_output_rgb'] if 'stack_output_rgb' in base
               else base['grade_output_rgb'] if 'grade_output_rgb' in base
               else fallback_rgb)
    # Measure decoded output, including the same domain guards as the grade.
    # Share this appearance conversion with neutral statistics on the adaptance page.
    appearance=(_shared['appearance'] if _shared is not None else
                encode_cones(model.tone.cones(after_rgb),model.tone.reference))
    after=appearance.q if appearance is not None else _shared['after']
    energy=appearance.ja*np.abs(appearance.ja) if _shared is None else _shared['energy']
    saturation=project(after,energy) if _shared is None else _shared['saturation']
    bounded_saturation=bounded_project(after,energy) if _shared is None else _shared['bounded_saturation']
    stats=None
    waveform=output_waveform(after,selected['weights'],model.tone.plot_max*100,x=selected.get('sample_x'))
    cloud=vectorscope(after_rgb,selected['weights'],space)
    if need_neutral:
        ids=selected['sample_indices']
        stats=robust.describe_sample(saturation.reshape(-1,2)[ids],min_std=0.);stats['sample_indices']=ids
    if _shared is None:selected['_input'].pop('prepared',None)
    return dict(selected,model=model,after=after,after_energy=energy,saturation=saturation,
        bounded_saturation=bounded_saturation,neutral=stats,
        saturation_excluded=int(np.count_nonzero(~np.isfinite(saturation).all(axis=-1)&(selected['weights']>0))),
        waveform_selected=waveform['density'],waveform_outside=waveform['outside'],
        vectorscope=cloud,label=base.get('label','Input'))


def evaluate_all(base,params,space,cloud_points=True):
    """One appearance conversion and input analysis, four immutable selections.

    Focus and page selection are presentation state. Every completed bundle
    contains all tones, including neutral statistics needed by the Warp page.
    """
    model=ch.model(base['model'].params,params,base['model'].space)
    cached=base.get('_input_cache')
    direct='stack_output_rgb' not in base and 'grade_output_rgb' not in base
    data=input_analysis(base,model,cached,prepare=not direct)
    guarded=None
    if direct:
        from ..model.forward import Model
        engine=Model(model.tone,model)
        guarded=(None if not (model.tone.active or model.active)
                  else engine.guarded_appearance(base['stage2_appearance']))
        if guarded is None:
            from ..model.pipeline import Prepared
            rgb=Prepared(base,model).apply(model)
            appearance=encode_cones(model.tone.cones(rgb),model.tone.reference)
            after=appearance.q;energy=appearance.ja*np.abs(appearance.ja)
        else:
            after,energy,native=guarded
            rgb=engine.from_native(native)
            appearance=None
    else:
        rgb=(base['stack_output_rgb'] if 'stack_output_rgb' in base else base['grade_output_rgb'])
        appearance=encode_cones(model.tone.cones(rgb),model.tone.reference)
        after=appearance.q
        energy=appearance.ja*np.abs(appearance.ja)
    shared=dict(input=data,rgb=rgb,appearance=appearance,after=after,energy=energy,
        saturation=project(after,energy),
        bounded_saturation=bounded_project(after,energy) if cloud_points else None,
        selections=None if cached is None else cached.get('selections'))
    result=[]
    for tone in range(4):
        shared_tone=dict(shared,selection=None if shared['selections'] is None else shared['selections'][tone])
        result.append(evaluate(base,params,space,tone,True,model,shared_tone))
    result=tuple(result)
    data.pop('prepared',None)
    return result

