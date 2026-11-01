"""Helper-owned, demand-driven Stage 3 scopes with compact display receipts.

Base registrations retain float64 Stage 2 values and a single exclusive group
ID. Output arrays and GPU resources never appear in the returned display DTO.
"""
from collections import OrderedDict
import time
from ...perceptual_grade_core.model import color
from . import scope_cloud_gpu as cloud
from ...perceptual_grade_core.model.scope_active import SCHEMA, _check, demands, _rgba, prepare, group, _rec709, _cpu_output, density_levels, _cpu_jc, _histograms, _views, _cpu_warp, project
from ...perceptual_grade_core.model import statistics as robust




def output(base,snapshot,index,use_gpu,checkpoint=None):
    model=color.model(snapshot['tone'],snapshot['params'],snapshot['grading_space'])
    key=(repr(snapshot['params']),repr(snapshot.get('_scope_evaluation',())),index,bool(use_gpu))
    if key in base['outputs']:
        base['outputs'].move_to_end(key)
        return model,base['groups'][index],base['outputs'][key]
    selected=group(base,model,index,checkpoint)
    if use_gpu:
        from . import scope_active_gpu
        try:out=scope_active_gpu.evaluate(base,model,selected,checkpoint)
        except cloud.UnsupportedGPUCloud as exc:
            out=_cpu_output(base,model,selected,checkpoint);out['fallback_reason']=str(exc)
    else:out=_cpu_output(base,model,selected,checkpoint)
    out.update(statistics=None,rasters=OrderedDict(),jc=None)
    base['outputs'][key]=out
    while len(base['outputs'])>8:base['outputs'].popitem(last=False)
    return model,selected,out

def _visual(base,snapshot,use_gpu=True,checkpoint=None):
    """Calculate only demanded pages/tones; return textures and small metadata."""
    started=time.perf_counter();colors={};rasters=[];backends=set();pending=[]
    for index,warp in demands(snapshot):
        _check(checkpoint)
        model,selected,out=output(base,snapshot,index,use_gpu,checkpoint)
        backend=out['backend'];backends.add(backend)
        item=dict(model=model,label=base['label'],_active_tone=index,
                  _active_page='warp' if warp else 'jc',backend=backend)
        if warp:
            records=[]
            for view in _views(snapshot,index):
                key=repr(view)
                if key not in out['rasters']:
                    if backend=='gpu':
                        from . import scope_active_gpu
                        record=scope_active_gpu.warp(out,view,index,checkpoint)
                    else:
                        record=_cpu_warp(out,view,index,checkpoint)
                    out['rasters'][key]=record
                    while len(out['rasters'])>8:out['rasters'].popitem(last=False)
                out['rasters'].move_to_end(key)
                records.append(out['rasters'][key])
            rasters.extend(records)
            item.update(_warp_clouds=tuple(records),_warp_tone=index,neutral=out['statistics'],
                        statistics_status='ready' if out['statistics'] is not None else 'pending')
            if out['comparison'] is not None:item['_comparison']=dict(available=True)
            if out['statistics'] is None:pending.append(index)
        else:
            if out['jc'] is None:
                if backend=='gpu':
                    from . import scope_active_gpu
                    out['jc']=scope_active_gpu.jc(out,model,checkpoint)
                else:
                    out['jc']=_cpu_jc(out,model)
                    if out['comparison'] is not None:out['jc']['_comparison']=_cpu_jc(out['comparison'],model)
            item.update(out['jc'])
            if selected['histograms'] is None:
                if backend=='gpu':
                    from . import scope_active_gpu
                    selected['histograms']=scope_active_gpu.histograms(selected,model,checkpoint)
                else:selected['histograms']=_histograms(selected,model)
            item.update(histograms={name:selected['histograms'][name] for name in ('low','high')},
                        sectors=selected['histograms']['sectors'])
        colors[f'{index}:{"warp" if warp else "jc"}']=item
    return dict(model=base['model'],label=base['label'],_active_colors=colors,
        _warp_clouds=tuple(rasters),_active_statistics_pending=tuple(pending),
        _scope_key=snapshot['key'],_scope_provenance=snapshot.get('_scope_provenance',()),
        _scope_evaluation=snapshot.get('_scope_evaluation',()),
        _active_backend='gpu' if backends=={'gpu'} else 'cpu',
        _active_visual_seconds=time.perf_counter()-started)


def visual(base,snapshot,use_gpu=True,checkpoint=None):
    """Expected device range limits use local CPU textures, preserving health."""
    try:return _visual(base,snapshot,use_gpu,checkpoint)
    except cloud.UnsupportedGPUCloud as exc:
        if not use_gpu:raise
        result=_visual(base,snapshot,False,checkpoint)
        result['_active_fallback_reason']=str(exc)
        return result


def statistics(base,snapshot,checkpoint=None):
    """Gather at most 512 linear coordinates per demanded Warp tone, then fit."""
    result={}
    for index,warp in demands(snapshot):
        if not warp:continue
        _check(checkpoint)
        matches=[out for key,out in base['outputs'].items()
                 if key[:3]==(repr(snapshot['params']),repr(snapshot.get('_scope_evaluation',())),index)]
        if not matches:raise ValueError('Scope statistics require a matching visual output')
        out=matches[-1];selected=base['groups'][index]
        if out['statistics'] is None:
            if out['backend']=='gpu':
                from . import scope_active_gpu
                points=scope_active_gpu.gather(out,selected['sample_ids'],checkpoint)
            else:points=project(out['q'][selected['sample_ids']],out['energy'][selected['sample_ids']])
            out['statistics']=robust.describe_sample(points,min_std=0.,checkpoint=checkpoint)
        result[str(index)]=dict(neutral=out['statistics'],statistics_status='ready')
    return dict(_active_statistics=result)
