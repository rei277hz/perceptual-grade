"""GPU selection and foreground status; resources belong to the runtime."""

from ..integration.components import Component

def create(_application):
    from dataclasses import asdict
    import json

    RemoteBaker = _application.component('jobs.baker').RemoteBaker
    from ..jobs.configuration import GPU_ROLES
    current_runtime = _application.component('services').current_runtime; get_runtime = _application.component('services').get_runtime

    _last_status = None


    def status(role='gpu'):
        if role not in GPU_ROLES:raise ValueError('Unknown GPU role: '+str(role))
        runtime = current_runtime()
        if runtime is None or runtime.closed:
            return dict(state='ready', reason='', diagnostic=None)
        return runtime.gpu_status(role=role)


    def details(role='gpu'):
        diagnostic = status(role=role)['diagnostic']
        return '' if diagnostic is None else json.dumps(asdict(diagnostic), indent=2, ensure_ascii=False)


    def _status_tick():
        nonlocal _last_status
        runtime = current_runtime()
        if runtime is None or runtime.closed or runtime.fenced:return None
        scheduler = _application.component('graph.scheduler')
        if not scheduler.lifecycle_admits('discover'):return .05
        runtime.poll()
        current = tuple(runtime.gpu_status(role=role) for role in GPU_ROLES)
        identity = tuple((item['state'], item['reason'], item['diagnostic']) for item in current)
        if identity != _last_status:
            _last_status = identity
            redraw = _application.component('graph.lut').redraw
            redraw()
        return .05 if runtime.jobs else None


    _status_tick = _application.component('integration.outcomes').callback(
        _status_tick, control='gpu', recovery='perceptual_grade.retry_recovery')

    def _watch():
        import bpy
        if not bpy.app.background and not bpy.app.timers.is_registered(_status_tick):
            bpy.app.timers.register(_status_tick, first_interval=.05)


    def acquire(model, progressive, encoded, stack=False):
        _watch()
        return RemoteBaker(model, progressive, encoded, stack, lane='gpu'), None


    def retry(role='gpu'):
        handle = get_runtime().retry_gpu(role=role)
        _watch()
        return handle


    def clear():
        """Release the UI timer without resetting runtime health or valid outputs."""
        nonlocal _last_status
        import bpy
        if bpy.app.timers.is_registered(_status_tick):bpy.app.timers.unregister(_status_tick)
        _last_status = None

    def _set_export__last_status(value):
        nonlocal _last_status
        _last_status = value

    def _set_export_status(value):
        nonlocal status
        status = value

    def _set_export_details(value):
        nonlocal details
        details = value

    def _set_export__status_tick(value):
        nonlocal _status_tick
        _status_tick = value

    def _set_export__watch(value):
        nonlocal _watch
        _watch = value

    def _set_export_acquire(value):
        nonlocal acquire
        acquire = value

    def _set_export_retry(value):
        nonlocal retry
        retry = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_current_runtime(value):
        nonlocal current_runtime
        current_runtime = value

    def _set_export_get_runtime(value):
        nonlocal get_runtime
        get_runtime = value

    return Component(readers={
        "__name__": lambda: __name__,
        'GPU_ROLES': lambda: GPU_ROLES,
        'RemoteBaker': lambda: RemoteBaker,
        '_last_status': lambda: _last_status,
        '_status_tick': lambda: _status_tick,
        '_watch': lambda: _watch,
        'acquire': lambda: acquire,
        'asdict': lambda: asdict,
        'clear': lambda: clear,
        'current_runtime': lambda: current_runtime,
        'details': lambda: details,
        'get_runtime': lambda: get_runtime,
        'json': lambda: json,
        'retry': lambda: retry,
        'status': lambda: status,
    }, writers={
        '_last_status': _set_export__last_status,
        'current_runtime': _set_export_current_runtime,
        'get_runtime': _set_export_get_runtime,
    'status': _set_export_status, 'details': _set_export_details, '_status_tick': _set_export__status_tick, '_watch': _set_export__watch, 'acquire': _set_export_acquire, 'retry': _set_export_retry, 'clear': _set_export_clear})
