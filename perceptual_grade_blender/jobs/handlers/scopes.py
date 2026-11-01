"""Helper scope jobs; caches and numerical objects never leave this process."""
from collections import OrderedDict
from concurrent.futures import CancelledError
import hashlib
import time
from ....perceptual_grade_core.jobs.codec import thaw_scope_request, freeze_scope_result, freeze_full_scope_result
from ....perceptual_grade_core.model import scope_analysis

_cache = OrderedDict()
_warp_results = OrderedDict()
_sources = OrderedDict()
_active_bases = OrderedDict()
_SOURCE_LIMIT = 8
_SOURCE_BYTES = 64 * 1024 * 1024


def _backend(context):
    return dict(backend=context.role, generation=getattr(context, 'generation', 0),
                session=getattr(context, 'session', ''))


def _control(kind, cache_id, context):
    return freeze_scope_result(dict(_scope_control=kind, cache_id=cache_id,
                                    **_backend(context)))


def _source_bytes(value, seen=None):
    """Bound retained source data, counting aliased numerical arrays once."""
    import numpy as np
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from ....perceptual_grade_core.model.lut import Cube
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,Cube):
        return _source_bytes(value.values,seen)+_source_bytes(getattr(value,'prepared_pixels',None),seen)
    if isinstance(value,Mapping):return sum(_source_bytes(v,seen) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(_source_bytes(v,seen) for v in value)
    if is_dataclass(value):return sum(_source_bytes(getattr(value,f.name),seen) for f in fields(value))
    return 0


def register(inputs, context):
    """Acknowledge a helper-owned seed before accepting array-free updates.

    Cache IDs are local to this helper generation. A runtime retry on another
    backend cannot accidentally resolve the original helper's source.
    """
    from ....perceptual_grade_core.model.scope_color import _immutable
    context.checkpoint()
    snapshot=thaw_scope_request(inputs)
    cache_id=snapshot['cache_id']
    if not isinstance(cache_id,str) or not cache_id:
        raise ValueError('Scope source cache ID must be a nonempty string')
    size=_source_bytes(snapshot)
    if size>_SOURCE_BYTES:raise ValueError('Scope source exceeds bounded seed cache')
    # Own the data independently of transport input-file cleanup. Never retain
    # a request-local publication or a mutable caller-owned array as a seed.
    seed=_immutable(snapshot)
    context.checkpoint()
    _sources[cache_id]=(seed,size,_backend(context))
    _sources.move_to_end(cache_id)
    while len(_sources)>_SOURCE_LIMIT or sum(v[1] for v in _sources.values())>_SOURCE_BYTES:
        _sources.popitem(last=False)
    return _control('ready',cache_id,context)


def _request_identity(snapshot, context):
    """Fence phased publications, including recovery onto a different helper."""
    identity=(snapshot['cache_id'],snapshot['base_key'],snapshot['key'],
              snapshot.get('_scope_demands',()),snapshot.get('_warp_views',()),
              snapshot.get('_scope_evaluation',()),snapshot['_scope_provenance'],
              tuple(_backend(context).items()))
    return hashlib.sha256(repr(identity).encode('utf8')).hexdigest()


def update(inputs, context):
    """Publish active rasters first, then only their matching p95 summary."""
    from ...model import scope_active
    context.checkpoint()
    request=thaw_scope_request(inputs)
    cache_id=request['cache_id']
    entry=_sources.get(cache_id)
    if entry is None or entry[2]!=_backend(context):
        # This is normal after restart, eviction or GPU-to-CPU retry. Raising
        # would turn a cache miss into a device failure and retry without seed.
        return _control('needs_seed',cache_id,context)
    _sources.move_to_end(cache_id)
    snapshot=dict(entry[0]);snapshot.update(request)
    snapshot['_scope_source_cache_id']=cache_id
    identity=_request_identity(snapshot,context)
    started=time.perf_counter()
    valid=all(size==65 for _,size in snapshot['_scope_provenance'])
    base=scope_active.prepare(snapshot,_active_bases if valid else OrderedDict(),context.checkpoint)
    prepared=time.perf_counter()
    use_gpu=context.role=='scope_gpu'
    if use_gpu:
        from . import bake
        bake._device()
    visual=scope_active.visual(base,snapshot,use_gpu=use_gpu,checkpoint=context.checkpoint)
    visual=dict(visual,_scope_part='visual',_scope_request=identity,
                _scope_key=snapshot['key'],_scope_evaluation=snapshot.get('_scope_evaluation',()),
                _scope_provenance=snapshot['_scope_provenance'],
                _scope_backend=_backend(context),
                _scope_timings=dict(prepare_seconds=prepared-started,
                                    visual_seconds=time.perf_counter()-prepared))
    context.checkpoint()
    needs_stats=any(bool(warp) for _,warp in snapshot.get('_scope_demands',()))
    if not needs_stats:return freeze_scope_result(visual)
    context.preview(freeze_scope_result(visual))
    stats_started=time.perf_counter()
    try:
        statistics=scope_active.statistics(base,snapshot,checkpoint=context.checkpoint)
        delta=dict(statistics,_scope_stats_status='ready')
    except CancelledError:
        raise
    except Exception as error:
        # The visual remains valid when a CPU fit fails. Do not disable the
        # scope GPU or retransmit its rasters for this statistics-only error.
        from ....blender_runtime.execution.diagnostics import text
        delta=dict(_active_statistics={},_scope_stats_status='unavailable',
                   _scope_stats_error=text(error))
    context.checkpoint()
    delta.update(_scope_part='stats',_scope_request=identity,
                 _scope_key=snapshot['key'],_scope_provenance=snapshot['_scope_provenance'],
                 _scope_evaluation=snapshot.get('_scope_evaluation',()),
                 _scope_backend=_backend(context),
                 _scope_timings=dict(statistics_seconds=time.perf_counter()-stats_started))
    return freeze_scope_result(delta)


def probe(inputs, context):
    """Uncached scope conformance; LUT conformance cannot validate this path."""
    import numpy as np
    from ...model import scope_active, scope_active_gpu
    from ....perceptual_grade_core.model import linear as cm, tone, color, saturation
    from . import bake
    context.checkpoint()
    device=bake._device()
    space=cm.GradeSpace('D65')
    rgba=np.ones((8,8,4),dtype=float)
    rgba[...,:3]=np.random.default_rng(413).uniform(.04,.7,(8,8,3))
    view=dict(key=('scope-probe',),width=32,height=32,rect=(0.,0.,32.,32.),
              scale=1.,bounds=((-2.,-2.,2.,2.),)*4,tone=1)
    snapshot=dict(base_key=('scope-probe',),key=('scope-probe',),rgba=rgba,chain=(),
        label='Probe',signal=None,comparison=None,stack=False,space=space,grading_space=space,
        params=color.Parameters(global_gain=.1),tone=tone.Parameters(),view=(1,True),stage=3,
        linear=cm.Parameters(),matrix=np.eye(3),to_grade=np.eye(3),_scope_provenance=(),
        _scope_evaluation=('scope-probe',),
        _scope_demands=((1,True),(1,False),(3,True),(3,False)),
        _warp_views=(view,dict(view,key=('scope-probe-empty',),tone=3)))
    base=scope_active.prepare(snapshot,OrderedDict(),context.checkpoint)
    result=scope_active.visual(base,snapshot,use_gpu=True,checkpoint=context.checkpoint)
    if (result.get('_active_backend')!='gpu' or not result.get('_warp_clouds')
            or any(r.get('backend')!='gpu' for r in result['_warp_clouds'])):
        raise ArithmeticError('Scope conformance probe did not render on GPU')
    # Independent CPU bases prevent cached GPU histograms from validating
    # themselves. Exercise both a nonempty selected group and an empty one.
    cpu_base=scope_active.prepare(snapshot,OrderedDict(),context.checkpoint)
    expected=scope_active.visual(cpu_base,snapshot,use_gpu=False,checkpoint=context.checkpoint)
    if not len(base['groups'][1]['ids']) or len(base['groups'][3]['ids']):
        raise ArithmeticError('Scope conformance fixture lost its nonempty/empty groups')
    for index in (1,3):
        selected=result['_active_colors'][f'{index}:jc']
        oracle=expected['_active_colors'][f'{index}:jc']
        for field in ('waveform_levels','vectorscope_levels'):
            array=np.asarray(selected[field]);reference=np.asarray(oracle[field])
            if array.dtype!=np.uint8 or np.max(array,initial=0)>7 or array.shape!=reference.shape:
                raise ArithmeticError('Scope conformance density quantization failed')
            error=np.abs(array.astype(int)-reference.astype(int))
            if np.count_nonzero(error>1)/error.size>.005:
                raise ArithmeticError('Scope conformance density differs from CPU')
            if not index==3 and not np.any(array):
                raise ArithmeticError('Scope conformance nonempty density is blank')
            if index==3 and np.any(array):
                raise ArithmeticError('Scope conformance empty density is nonzero')
        for driver in ('low','high'):
            np.testing.assert_allclose(selected['histograms'][driver],oracle['histograms'][driver],
                                       rtol=8e-4,atol=5e-5)
        sectors=np.asarray(selected['sectors'])
        if sectors.dtype!=np.float32 or sectors.shape!=(48,) or not np.isfinite(sectors).all():
            raise ArithmeticError('Scope conformance hue sectors format failed')
        np.testing.assert_allclose(sectors,oracle['sectors'],rtol=8e-4,atol=5e-5)
        _,group,gpu_output=scope_active.output(base,snapshot,index,True,context.checkpoint)
        _,_,cpu_output=scope_active.output(cpu_base,snapshot,index,False,context.checkpoint)
        ids=np.arange(min(8,len(group['ids'])),dtype=np.int32)
        points=scope_active_gpu.gather(gpu_output,ids,context.checkpoint)
        oracle_points=saturation.project(cpu_output['q'][ids],cpu_output['energy'][ids])
        np.testing.assert_allclose(points,oracle_points,rtol=5e-4,atol=2e-5,equal_nan=True)
        if points.shape!=(len(ids),2) or points.nbytes>4096:
            raise ArithmeticError('Scope conformance statistical gather is unbounded')
    for actual,oracle in zip(result['_warp_clouds'],expected['_warp_clouds'],strict=True):
        pixels=np.asarray(actual['pixels']);reference=np.asarray(oracle['pixels'])
        if pixels.dtype!=np.uint8 or pixels.shape!=reference.shape:
            raise ArithmeticError('Scope conformance cloud format differs from CPU')
        error=np.abs(pixels.astype(int)-reference.astype(int))
        if np.count_nonzero(error>1)/error.size>.005:
            raise ArithmeticError('Scope conformance cloud differs from CPU')
        if (actual['tone']==3 and np.any(pixels)) or (actual['tone']==1 and not np.any(pixels)):
            raise ArithmeticError('Scope conformance cloud membership failed')
    context.checkpoint()
    return dict(conformance=True,device=device,backend='GPU',role=context.role,
                checks=('cloud','waveform','vectorscope','histogram','hue_sectors','linear_gather','empty_group'))


def linear(inputs, context):
    context.checkpoint()
    result = scope_analysis.evaluate_linear(thaw_scope_request(inputs))
    context.checkpoint()
    return freeze_scope_result(result)


def full(inputs, context):
    context.checkpoint()
    snapshot = thaw_scope_request(inputs)
    # The foreground validates the complete result against its publication
    # ticket. An invalid capture must not poison reusable Stage-1/2 analysis
    # while that validation rejects its envelope and the user retries.
    cache = _cache if all(size == 65 for _, size in snapshot['_scope_provenance']) else OrderedDict()
    result = scope_analysis.evaluate_snapshot(snapshot, cache)
    context.checkpoint()
    return freeze_full_scope_result(result)


def warp(inputs, context):
    """All CPU scopes stay authoritative; only Warp clouds use GPU pixels.

    A view-only request reuses the exact numerical result. Cache entries never
    contain view fields and never leave this helper; each publication gets its
    own key/provenance envelope. Bulk retries rasterize locally on CPU using the
    same texture-only display contract.
    """
    from ...model import scope_cloud_gpu
    context.checkpoint()
    started = time.perf_counter()
    snapshot = thaw_scope_request(inputs)
    valid = all(size == 65 for _, size in snapshot['_scope_provenance'])
    identity = repr((snapshot['base_key'], snapshot.get('_warp_analysis_key', snapshot['key']),
                     snapshot.get('_scope_evaluation'), snapshot['_scope_provenance']))
    result = _warp_results.get(identity) if valid else None
    reused = result is not None
    if result is None:
        # Cloud coordinates are produced on the GPU. Linear saturation remains
        # the existing CPU input for p95; every other CPU display is unchanged.
        numerical = dict(snapshot, _warp_gpu=True)
        result = scope_analysis.evaluate_snapshot(numerical, _cache if valid else OrderedDict())
        if valid:
            _warp_results[identity] = result
            while len(_warp_results) > 4:
                _warp_results.popitem(last=False)
    else:
        _warp_results.move_to_end(identity)
    cpu_seconds = time.perf_counter() - started
    context.checkpoint()
    if context.role == 'gpu' and snapshot.get('_warp_views'):
        from . import bake
        bake._device()
    clouds = scope_cloud_gpu.render(result, snapshot.get('_warp_views', ()),
                                   use_gpu=context.role == 'gpu', checkpoint=context.checkpoint)
    # Do not mutate cached backreferences to select a view or publication key.
    published = dict(result, _scope_key=snapshot['key'], _warp_clouds=clouds,
                     _scope_provenance=snapshot['_scope_provenance'],
                     _scope_evaluation=snapshot.get('_scope_evaluation', ()),
                     _warp_cpu_seconds=cpu_seconds, _warp_cpu_reused=reused)
    tone = snapshot['view'][0] if snapshot.get('view') else 0
    published['_color'] = result['_colors'][tone]
    if result.get('_comparisons_selected'):
        published['_comparison_selected'] = result['_comparisons_selected'][tone]
    context.checkpoint()
    return freeze_full_scope_result(published)
