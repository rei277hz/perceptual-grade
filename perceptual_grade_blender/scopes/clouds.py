"""Stage 3 page/tone demands, native texture views and session GPU selection."""

from ..integration.components import Component

def create(_application):
    import math
    import os
    from ...perceptual_grade_core.scopes.identity import ActiveScopeIdentity

    SCHEMA = 2
    _enabled = os.environ.get('PG_SCOPE_CLOUD_GPU', '').lower() in {'1', 'true', 'yes'}
    _views = {}
    _demands = {}


    def enabled():
        return _enabled


    def set_enabled(value):
        """Switch the experiment without persisting a preference in the blend file."""
        nonlocal _enabled
        value = bool(value)
        if value == _enabled:
            return
        _enabled = value
        # Backend changes retire worker captures, not compatible display textures.
        jobs = _application.component('scopes.jobs')
        jobs.refresh_backend()


    def descriptor(rect, scale, bounds, tone=None):
        """Enclose the pad in native pixels, retaining its subpixel clipping edge."""
        x, y, w, h = map(float, rect)
        scale = float(scale)
        bounds = tuple(tuple(map(float, row)) for row in bounds)
        if (len(bounds) != 4 or any(len(row) != 4 for row in bounds) or
                not all(math.isfinite(v) for v in (x, y, w, h, scale, *sum(bounds, ()))) or
                min(w, h, scale) <= 0 or any(a >= c or b >= d for a, b, c, d in bounds)):
            raise ValueError('Invalid Warp raster view')
        width = math.ceil(x + w) - math.floor(x)
        height = math.ceil(y + h) - math.floor(y)
        local = (x - math.floor(x), y - math.floor(y), w, h)
        key = (SCHEMA, width, height, local, scale, bounds)
        result=dict(key=key, width=width, height=height, rect=local, scale=scale, bounds=bounds)
        if tone is not None:
            if not isinstance(tone,int) or not 0<=tone<4:
                raise ValueError('Invalid Warp raster tone')
            result['tone']=tone
            result['key']=(*key,tone)
        return result


    def capture_view(node, box):
        """Called by the existing visible-editor scan; never retain native owners."""
        if box is None or node.pg_chromatic.family != 'WARP':
            return None
        nodes = _application.component('graph.nodes')
        warp = _application.component('ui.warp'); warp_view = _application.component('ui.warp_view')
        active = node.pg_chromatic.active_tone
        held = warp_view.radius(box.get('_warp_editor'), node, active)
        if held is None:
            model = nodes.color_model(node, nodes.grading_space(node))
            held = warp.view_radius(model, active)
        bound=(-held,-held,held,held)
        # The raster/UI compatibility contract indexes one of four bounds. The
        # tone tag ensures only this demanded group consumes this native view.
        return descriptor(box['ab'],box['scale'],(bound,)*4,tone=active)


    def replace_views(captured):
        """Replace the visible set atomically, deduplicating equal native rasters."""
        nonlocal _views
        _views = {}
        for pointer, values in captured.items():
            unique = {value['key']: value for value in values if value is not None}
            _views[pointer] = tuple(unique[key] for key in sorted(unique))


    def views(node):
        """Read value-only descriptors without scanning editors during UI draws."""
        return _views.get(node.as_pointer(), ())


    def replace_demands(captured):
        """Union visible page/tone consumers before any request is described."""
        nonlocal _demands
        _demands = {pointer: tuple(sorted(set(values))) for pointer, values in captured.items()}


    def demands(node, view=()):
        values=set(_demands.get(node.as_pointer(), ()))
        if len(view)==2 and isinstance(view[0],int) and isinstance(view[1],bool):
            values.add((int(view[0]),bool(view[1])))
        return tuple(sorted(values))


    def clear():
        _views.clear()
        _demands.clear()


    def decorate(spec, node):
        """Raster geometry changes the publication key, not cached CPU statistics."""
        view = spec.get('view', ())
        if spec.get('stage') != 3 or len(view) != 2 or not isinstance(view[0],int):
            return spec
        captured = views(node)
        requested = demands(node, view)
        spec['_scope_demands'] = requested
        spec['_scope_backend'] = enabled()
        spec['_warp_analysis_key'] = spec['key'].analysis()
        spec['_warp_views'] = captured
        spec['key'] = spec['key'].with_presentation(ActiveScopeIdentity(
            schema=SCHEMA, requested=requested,
            views=tuple(v['key'] for v in captured), enabled=enabled()))
        spec['_scope_evaluation'] = (*spec['_scope_evaluation'], ('active-scopes', SCHEMA))
        return spec

    def _set_export_SCHEMA(value):
        nonlocal SCHEMA
        SCHEMA = value

    def _set_export__demands(value):
        nonlocal _demands
        _demands = value

    def _set_export__enabled(value):
        nonlocal _enabled
        _enabled = value

    def _set_export__views(value):
        nonlocal _views
        _views = value

    def _set_export_enabled(value):
        nonlocal enabled
        enabled = value

    def _set_export_set_enabled(value):
        nonlocal set_enabled
        set_enabled = value

    def _set_export_descriptor(value):
        nonlocal descriptor
        descriptor = value

    def _set_export_capture_view(value):
        nonlocal capture_view
        capture_view = value

    def _set_export_replace_views(value):
        nonlocal replace_views
        replace_views = value

    def _set_export_views(value):
        nonlocal views
        views = value

    def _set_export_replace_demands(value):
        nonlocal replace_demands
        replace_demands = value

    def _set_export_demands(value):
        nonlocal demands
        demands = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_decorate(value):
        nonlocal decorate
        decorate = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ActiveScopeIdentity': lambda: ActiveScopeIdentity,
        'SCHEMA': lambda: SCHEMA,
        '_demands': lambda: _demands,
        '_enabled': lambda: _enabled,
        '_views': lambda: _views,
        'capture_view': lambda: capture_view,
        'clear': lambda: clear,
        'decorate': lambda: decorate,
        'demands': lambda: demands,
        'descriptor': lambda: descriptor,
        'enabled': lambda: enabled,
        'math': lambda: math,
        'os': lambda: os,
        'replace_demands': lambda: replace_demands,
        'replace_views': lambda: replace_views,
        'set_enabled': lambda: set_enabled,
        'views': lambda: views,
    }, writers={
        'SCHEMA': _set_export_SCHEMA,
        '_demands': _set_export__demands,
        '_enabled': _set_export__enabled,
        '_views': _set_export__views,
    'enabled': _set_export_enabled, 'set_enabled': _set_export_set_enabled, 'descriptor': _set_export_descriptor, 'capture_view': _set_export_capture_view, 'replace_views': _set_export_replace_views, 'views': _set_export_views, 'replace_demands': _set_export_replace_demands, 'demands': _set_export_demands, 'clear': _set_export_clear, 'decorate': _set_export_decorate})
