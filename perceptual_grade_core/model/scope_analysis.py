"""Scope analysis on immutable numerical captures, independent of Blender."""
import logging
import numpy as np
from . import linear as cm, tone as tm, scope_grid


def evaluate_signal(rgb, signal):
    from . import lut
    cube, before, after = signal
    if cube is None:
        raise ValueError('Scope signal has no final LUT')
    return lut.tetrahedral(cube.values, np.asarray(rgb) @ before.T) @ after.T


def distribution(j,alpha,max_j,sample_grid=None):
    height,width=j.shape
    mass,xx=scope_grid.values(sample_grid,j.shape)
    visible=(alpha>0)&(mass>0)
    finite=np.isfinite(j)&visible
    inside=finite&(j>=0)&(j<=max_j)
    bins_x=np.minimum(191,(xx[inside]*192).astype(int))
    bins_y=np.minimum(159,(j[inside]*160/max_j).astype(int))
    density=np.zeros((160,192),dtype=np.float64)
    np.add.at(density,(bins_y,bins_x),mass[inside])
    hist=np.histogram(j[inside],bins=128,range=(0,max_j),weights=mass[inside])[0]
    return dict(density=density,histogram=hist,invalid=float(np.sum(mass[visible&~np.isfinite(j)])),
                above=float(np.sum(mass[finite&(j>max_j)])),below=float(np.sum(mass[finite&(j<0)])),
                count=float(np.sum(mass[visible])))


def apply_chain(rgb,chain,space):
    rgb=np.asarray(rgb,dtype=float)
    for step in chain:
        if step[0]=='BAKED':
            _,cube,before,matrix=step
            rgb=evaluate_signal(rgb,(cube,before,matrix))
            continue
        if step[0]=='GRADE':
            _,model,before,after=step
            rgb=model.apply(rgb@before.T)@after.T
            continue
        raise ValueError('Unsupported scope chain step')
    return rgb


def evaluate_samples(rgba,linear,tone,space,chain=(),selected_model=None,matrix=None,input_matrix=None,sample_grid=None):
    rgb=apply_chain(rgba[:,:,:3],chain,space)
    from ..model.forward import Model
    model=selected_model or tm.model(tone,space);engine=Model(model)
    incoming=rgb if input_matrix is None else rgb@np.asarray(input_matrix).T
    before=incoming@np.asarray(cm.matrices(linear,model.space)['rgb'] if matrix is None else matrix).T
    prepared=engine.prepare_input(before)
    before_q=engine.encode_cones(model.cones(prepared))
    # Stage 2 is the tone appearance handoff. Keep it raw and signed; export
    # guards and chromatic controls belong to Stage 3 and must not affect this
    # publication.
    signed=model.apply_appearance(before_q)
    magnitude=lambda q:np.hypot(q.ja,np.sqrt(66*np.hypot(q.q[...,1],q.q[...,2])))/100
    before_j=magnitude(before_q);after_j=magnitude(signed)
    mass,x=scope_grid.values(sample_grid,rgba.shape[:2])
    return dict(before=distribution(before_j,rgba[:,:,3],model.plot_max,sample_grid),
                after=distribution(after_j,rgba[:,:,3],model.plot_max,sample_grid),model=model,
                before_j=before_j,after_j=after_j,source_rgb=rgb,stage1_rgb=before,
                stage2_input_rgb=prepared,stage2_appearance=signed,alpha=rgba[:,:,3],
                sample_grid=sample_grid,sample_mass=mass,sample_x=x)


def evaluate_linear(snapshot):
    """Matrix, signed Rec.709 projection and histogram only; no appearance math.

    Prepared source RGB is already in grading space and is immutable when
    published. Both source signals can be reused by the next matrix edit.
    """
    from . import scope_color as color
    s=snapshot
    def prepared(source):
        if source is None or 'rgb' in source:return source
        rgba=source['rgba']
        rgb=apply_chain(rgba[...,:3],source['chain'],s['scene_space'])@source['to_grade'].T
        return dict(key=source['key'],rgb=rgb,alpha=rgba[...,3],label=source['label'],
            sample_grid=source.get('sample_grid'),_scope_provenance=source['_scope_provenance'])
    source=prepared(s['source']);comparison=prepared(s['comparison'])
    rgb=source['rgb']@s['matrix'].T
    if comparison is not None and 'vectorscope' not in comparison:
        # Match the full comparison scope's finite/alpha policy without its
        # appearance encoding, tone distribution or contribution analysis.
        alpha=np.where(np.isfinite(comparison['rgb']).all(axis=-1),np.clip(comparison['alpha'],0,1),0.)
        mass,_=scope_grid.values(comparison.get('sample_grid'),alpha.shape)
        comparison=dict(comparison,alpha=alpha,
            vectorscope=color.vectorscope(comparison['rgb'],alpha*mass,s['space']))
    mass,_=scope_grid.values(source.get('sample_grid'),source['alpha'].shape)
    return dict(_scope_key=s['key'],_scope_provenance=s['_scope_provenance'],
        _scope_evaluation=s.get('_scope_evaluation',()),_source=source,_comparison_source=comparison,
        source_rgb=source['rgb'],stage1_rgb=rgb,alpha=source['alpha'],label=source['label'],
        linear_vectorscope=color.vectorscope(rgb,source['alpha']*mass,s['space']),_comparison=comparison)


def evaluate_snapshot(snapshot,cache):
    """Pure numerical analysis; cache ownership stays with the scope worker."""
    from . import scope_color as color, scope_reference as reference
    s=snapshot;grid=s.get('sample_grid')
    key=s['base_key'] if grid is None else (s['base_key'],scope_grid.key(grid))
    model=tm.model(s['tone'],s['grading_space'])
    if key not in cache:
        if s['stack']:
            incoming=apply_chain(s['rgba'][...,:3],s['chain'],s['space'])
            outgoing=evaluate_signal(incoming,s['signal'])
            incoming=incoming@s['to_grade'].T;outgoing=outgoing@s['to_grade'].T
            before_j=model.appearance(incoming)[...,0]/100;after_j=model.appearance(outgoing)[...,0]/100
            mass,x=scope_grid.values(grid,s['rgba'].shape[:2])
            base=dict(before=distribution(before_j,s['rgba'][...,3],model.plot_max,grid),
                after=distribution(after_j,s['rgba'][...,3],model.plot_max,grid),model=model,before_j=before_j,
                after_j=after_j,alpha=s['rgba'][...,3],stack_input_rgb=incoming,stack_output_rgb=outgoing,
                sample_grid=grid,sample_mass=mass,sample_x=x,
                label=s['label'],signature=s['signal'][0].signature if s['signal'][0] is not None else None)
        else:
            base=evaluate_samples(s['rgba'],s['linear'],s['tone'],s['space'],s['chain'],model,s['matrix'],s['to_grade'],grid)
            base['label']=s['label']
        rgb=base['stack_output_rgb'] if s['stack'] else base['stage1_rgb']
        base['linear_vectorscope']=color.vectorscope(rgb,base['alpha']*base['sample_mass'],s['grading_space'])
        # Input membership/capacity uses the colour model API; the tone model
        # above is retained for Stage 1/2 appearance and distributions. Build
        # this lazily for Stage 3 so a tone-only publication does no chromatic
        # analysis work.
        if s.get('stage') != 2:
            analysis_model=color.ch.model(s['tone'],s['params'],s['grading_space'])
            base['_input_cache']=color.input_analysis_cache(base,analysis_model)
        if len(cache)>=4:cache.popitem(last=False)
        cache[key]=color._immutable(base)
    if s.get('stage') != 2 and '_input_cache' not in cache[key]:
        # A tone-only publication intentionally leaves chromatic analysis out
        # of the worker entry. The first Stage 3 request promotes that same
        # Stage 1/2 record with the immutable input analysis.
        promoted=dict(cache[key])
        analysis_model=color.ch.model(s['tone'],s['params'],s['grading_space'])
        promoted['_input_cache']=color.input_analysis_cache(promoted,analysis_model)
        cache[key]=color._immutable(promoted)
    base=dict(cache[key]);base['_scope_key']=s['key'];base['_scope_provenance']=s['_scope_provenance']
    base['stage']=s.get('stage',3)
    if '_scope_evaluation' in s:
        base['_scope_evaluation']=s['_scope_evaluation']
    if s.get('stage') == 2:
        base['_comparison']=base['_comparison_selected']=None
        if s['comparison'] is not None:
            try:
                rgba,chain=s['comparison']
                base['_comparison']=reference.evaluate(rgba,chain,s['space'],base,s['to_grade'])
            except (ValueError,ArithmeticError,TypeError) as exc:
                logging.getLogger(__package__).warning('Color Reference scope: %s',''.join((str(exc),)))
        return base
    # Regular Grade scopes are analytical: their own installed cube is never
    # sampled.  Baked upstream steps remain represented in ``chain`` and are
    # evaluated before Stage 2 by ``evaluate_samples``.
    base['_colors']=color.evaluate_all(base,s['params'],s['grading_space'],
                                     cloud_points=not s.get('_warp_gpu',False))
    for selected in base['_colors']:selected['_base']=base
    base['_color']=base['_colors'][s['view'][0] if s['view'] else 0]
    base['_comparison']=base['_comparison_selected']=None
    if s['comparison'] is not None:
        try:
            rgba,chain=s['comparison']
            base['_comparison']=reference.evaluate(rgba,chain,s['space'],base,s['to_grade'],
                                                   cloud_points=not s.get('_warp_gpu',False))
            base['_comparisons_selected']=tuple(reference.select(base['_comparison'],selected,model.plot_max*100)
                for selected in base['_colors'])
            base['_comparison_selected']=base['_comparisons_selected'][s['view'][0] if s['view'] else 0]
        except (ValueError,ArithmeticError,TypeError) as exc:
            logging.getLogger(__package__).warning('Color Reference scope: %s',''.join((str(exc),)))
    return base

