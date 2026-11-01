"""GPU overlay; drawing and interaction are separate from image evaluation."""

from ..integration.components import Component

def create(_application):
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text
    remove_node_group = _application.app.blender.native.remove_node_group; remove_timer = _application.app.blender.native.remove_timer
    native = _application.component('localization.i18n').native; M = _application.component('localization.i18n').M
    from ...blender_runtime.execution.diagnostics import log_exception
    tr = _application.component('localization.i18n').tr; display = _application.component('localization.i18n').display; error_text = _application.component('localization.i18n').error_text; CONTEXT = _application.component('localization.i18n').CONTEXT; marker = _application.component('localization.i18n').marker

    import math
    import logging
    import re
    import time
    import unicodedata
    import bpy
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve; OwnedField = _application.component('ownership').OwnedField
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader
    from mathutils import Matrix
    from ...perceptual_grade_core.model import linear as cm
    nodes = _application.component('graph.nodes')
    editing = _application.component('graph.editing')
    hit_handle = _application.component('ui.geometry').hit_handle; in_rect = _application.component('ui.geometry').in_rect; diagram_layout = _application.component('ui.geometry').diagram_layout; tone_layout = _application.component('ui.geometry').tone_layout; exposure_point = _application.component('ui.geometry').exposure_point; balance_point = _application.component('ui.geometry').balance_point; gain_drag = _application.component('ui.geometry').gain_drag; FINE_GAIN = _application.component('ui.geometry').FINE_GAIN
    theme = _application.component('ui.theme')
    locus = _application.component('ui.locus')
    gizmo_help = _application.component('ui.help')
    warp_view = _application.component('ui.warp_view')

    LOG = logging.getLogger(__package__)
    _handle = None
    _registered = False
    _classes = []
    _routers = {}
    _hover_cursors = {}
    _shaders = {}
    _last_error = ''
    _drag_timers={}
    _active_drags={}
    COLORS = theme.COLORS
    TEXT = theme.TEXT
    MUTED = theme.MUTED


    def shader(name):
        if name not in _shaders:
            _shaders[name] = gpu.shader.from_builtin(name)
        return _shaders[name]


    def lines(coords, color, width=1):
        if not coords:
            return
        sh = shader('POLYLINE_UNIFORM_COLOR')
        batch = batch_for_shader(sh, 'LINES', {'pos': [(x, y, 0) for x, y in coords]})
        sh.bind();sh.uniform_float('viewportSize', gpu.state.viewport_get()[2:])
        sh.uniform_float('lineWidth', width);sh.uniform_float('color', color)
        batch.draw(sh)


    def triangles(coords, color):
        if not coords:
            return
        sh = shader('UNIFORM_COLOR')
        batch=getattr(coords,'batch',None)
        if batch is None:
            batch = batch_for_shader(sh, 'TRIS', {'pos': [(x, y, 0) for x, y in coords]})
            if isinstance(coords,CachedTriangles):coords.batch=batch
        sh.bind();sh.uniform_float('color', color);batch.draw(sh)


    class CachedTriangles(list):
        """Immutable scope geometry may retain its GPU vertex batch between draws."""
        batch=None



    class ColoredMesh:
        def __init__(self,coords,colors):
            self.coords=coords;self.colors=colors;self.batch=None


    def colored_triangles(mesh):
        sh=shader('SMOOTH_COLOR')
        if mesh.batch is None:
            mesh.batch=batch_for_shader(sh,'TRIS',{'pos':[(x,y,0) for x,y in mesh.coords],'color':mesh.colors})
        sh.bind();mesh.batch.draw(sh)

    def flush_color(op):
        if getattr(op,'_pending_color',None):
            nodes.write_chromatic(op._node,op._pending_color)
            op._pending_color=None;op._last_commit=time.monotonic()


    def end_color_drag(op,commit=True):
        _active_drags.pop(id(op), None)
        jobs = _application.component('scopes.jobs')
        scheduler = _application.component('graph.scheduler')
        warp = _application.component('ui.warp')
        try:
            if commit:flush_color(op)
            op._pending_color=None
            scheduler.end(getattr(op,'_schedule_gesture',None))
            op._schedule_gesture=None
            # Capture the released settings with their settled bounds. The display
            # retains/reprojects the preceding raster until this view is rendered.
            if getattr(op,'_warp_fit_token',None) is not None:
                warp_view.end(op._warp_fit_token)
            op._warp_fit_token=None
            jobs._sync_consumers()
            if commit:jobs.finish_edit(op._node)
        finally:
            scheduler.end(getattr(op,'_schedule_gesture',None))
            op._schedule_gesture=None
            op._pending_color=None
            # A settled gesture is a mouse release, never a 60 Hz commit or pause.
            if getattr(op,'_warp_fit_token',None) is not None:
                warp_view.end(op._warp_fit_token)
            op._warp_fit_token=None
            warp.end_scale_drag(getattr(op,'_warp_scale_token',None))
            op._warp_scale_token=None
            warp.end_drag(getattr(op,'_warp_dot_token',None))
            op._warp_dot_token=None
            pair=_drag_timers.pop(id(op),None)
            if pair:
                wm,timer=resolve(pair[0]),resolve(pair[1])
                try:
                    if wm is not None and timer is not None:remove_timer(wm,timer)
                except (ReferenceError,RuntimeError):pass
            op._drag_timer=None


    def keyframe_drag(op, context):
        """Release and save commit the same accepted parameter values."""
        if context is None or not context.scene.tool_settings.use_keyframe_insert_auto:
            return
        if op._mode == 'chromatic':
            from ...perceptual_grade_core.model import color as ch
            current = nodes.read_chromatic(op._node)
            for name in ch.PARAM_NAMES:
                if getattr(current, name) != getattr(op._chromatic_start, name):
                    nodes.keyframe_chromatic(op._node, name)
        elif op._mode in {'tone_level', 'tone_curve', 'fit'}:
            from ...perceptual_grade_core.model import tone as tm
            current = nodes.read_tone(op._node)
            for name in tm.PARAM_NAMES:
                if getattr(current, name) != getattr(op._tone_start, name):
                    nodes.keyframe_tone(op._node, name)
        elif op._mode in {'exposure', 'balance', 'white', 'scope_white', 'cr', 'yb'}:
            current = nodes.read_parameters(op._node)
            for name in cm.PARAM_NAMES:
                if getattr(current, name) != getattr(op._start, name):
                    nodes.keyframe_parameter(op._node, name)


    def release_drag_references(op):
        """A finished UNDO operator can outlive every RNA object in its gesture.

        Explicitly detach resource-bearing fields; keep small immutable numerical
        snapshots usable for diagnostics. This runs after commit/restore/keyframing.
        """
        _active_drags.pop(id(op), None)
        op._lifecycle_closed = True
        for name in ('_context', '_node', '_grade_start', '_area', '_wm', '_window',
                     '_drag_timer', '_white_drag', '_color_action', '_tone_model', '_box'):
            if hasattr(op, name):setattr(op, name, None)


    def finish_drag(op):
        """Exhaust cleanup even when an interrupted modal path has not settled."""
        try:
            if (id(op) in _active_drags or id(op) in _drag_timers or
                    any(getattr(op, name, None) is not None for name in
                        ('_schedule_gesture', '_warp_fit_token', '_warp_scale_token', '_warp_dot_token'))):
                end_color_drag(op, commit=False)
        finally:
            release_drag_references(op)


    def settle_drags(commit=False):
        """Fence modal objects before Blender replaces RNA or serializes it.

        Save accepts buffered edits. Load/undo/reload discard transient snapshots
        without restoring old values into newly loaded or already undone settings.
        """
        warp_edit = _application.component('ui.warp_edit')
        from ...blender_runtime.execution.diagnostics import Failure
        errors = []
        for op_ref in tuple(_active_drags.values()):
            op=resolve(op_ref)
            if op is None:continue
            op._lifecycle_closed = True
            try:
                end_color_drag(op, commit=commit)
                if commit:
                    keyframe_drag(op, getattr(op, '_context', None))
            except Exception as exc:
                errors.append(Failure.capture(exc))
                log_exception(LOG,'Could not settle a Perceptual Grade gesture')
            finally:
                _active_drags.pop(id(op), None)
                set_hover_cursor(getattr(op, '_window', None))
                release_drag_references(op)
        try:
            warp_edit.settle(commit=commit)
        except Exception as exc:
            errors.append(Failure.capture(exc))
            log_exception(LOG,'Could not settle a Perceptual Grade sidebar edit')
        if commit and errors:
            raise errors[0].exception('Could not settle a Perceptual Grade gesture')


    def cleanup_drag_timers():
        settle_drags(commit=False)
        for wm_ref,timer_ref in tuple(_drag_timers.values()):
            wm,timer=resolve(wm_ref),resolve(timer_ref)
            if wm is None or timer is None:continue
            try:remove_timer(wm,timer)
            except (ReferenceError,RuntimeError):pass
        _drag_timers.clear()
        warp_view.clear()


    def set_hover_cursor(window,cursor=None):
        """Own at most one modal cursor per window, restoring it on every exit."""
        if window is None:return
        try:
            key=window.as_pointer();old=_hover_cursors.get(key)
            if old is not None and old[1]==cursor:return
            if old is not None:
                _hover_cursors.pop(key,None);window.cursor_modal_restore()
            if cursor is not None:
                window.cursor_modal_set(cursor);_hover_cursors[key]=(retain(window,'diagram.hover_window'),cursor)
        except (ReferenceError,RuntimeError):pass


    def cleanup_hover_cursors():
        warp = _application.component('ui.warp')
        for window_ref,_ in tuple(_hover_cursors.values()):set_hover_cursor(resolve(window_ref))
        _hover_cursors.clear()
        gizmo_help.clear()
        warp_view.clear()
        warp.clear()


    def scope_status_target(context, node, box):
        """One status/Retry location shared by drawing and pointer routing."""
        stage = getattr(context.window_manager, 'pg_stage', 'LINEAR')
        if stage == 'REMAP':
            return box['before'], 'full'
        if stage == 'CHROMATIC':
            return box['ab'] if node.pg_chromatic.family == 'WARP' else box['waveform'], 'full'
        return box['vectorscope'], 'full' if editing.is_target(node) else 'linear'


    def scope_retry_hit(context, node, box, point):
        jobs = _application.component('scopes.jobs')
        gpu = _application.component('graph.gpu')
        color = _application.component('ui.color')
        rect, kind = scope_status_target(context, node, box)
        inputs=_application.component('scopes.sampling').scope_input_status(node,context)
        health=gpu.status(role=scope_gpu_role(context))
        if health['state']!='ready':
            x,y,w,h=rect
            gpu_rect=rect
            for action,button in color.gpu_status_rects(gpu_rect,box['scale']).items():
                if action=='gpu_retry' and health['state']=='probing':continue
                if action=='gpu_details' and health['diagnostic'] is None:continue
                if in_rect(point,button):return action
        return kind if (jobs.status(node, kind)['state'] == 'error' and
                       in_rect(point, color.scope_retry_rect(rect, box['scale']))) else None

    def scope_gpu_role(context):
        clouds = _application.component('scopes.clouds')
        manager=getattr(context,'window_manager',None)
        return 'scope_gpu' if getattr(manager,'pg_stage',None)=='CHROMATIC' and clouds.enabled() else 'gpu'


    def comparison_hit(context,node,box,point):
        """Toggle targets exclude every existing draggable/selectable handle."""
        comparison = _application.component('scopes.reference')
        if not comparison.connected(node):return False
        stage=getattr(context.window_manager,'pg_stage','LINEAR')
        if stage=='LINEAR':return in_rect(point,box['vectorscope'])
        space=nodes.grading_space(node)
        if stage=='CHROMATIC':
            chromatic_geometry = _application.component('ui.color_geometry')
            if getattr(node.pg_chromatic,'family','')=='WARP':
                chromatic_diagram = _application.component('ui.color')
                return (in_rect(point,chromatic_geometry.square(box['ab'])) and
                        chromatic_diagram.hit(node,space,box,point)['kind']=='consume')
            if in_rect(point,chromatic_geometry.square(box['vectorscope'])):return True
            if in_rect(point,box['waveform']):
                return chromatic_geometry.waveform_tone(point,box,nodes.color_model(node,space)) is None
        elif stage=='REMAP':
            tonal_diagram = _application.component('ui.tone')
            from ...perceptual_grade_core.model import tone as tonal_math
            model=tonal_math.model(nodes.read_tone(node),space)
            for side in ('before','after'):
                if in_rect(point,box[side]):return tonal_diagram.hit_level(point,box,model,side) is None
            if in_rect(point,box['chart'].rect):
                return tonal_diagram.hit_curve(point,box,model,context.window_manager.pg_selected_tone) is None
        return False


    def _redraw_rail_hover(window):
        if window is None:return
        try:
            for area in window.screen.areas:
                if area.type=='NODE_EDITOR':area.tag_redraw()
        except (AttributeError,ReferenceError,RuntimeError):pass


    def clear_rail_hover(window):
        warp = _application.component('ui.warp')
        if warp.clear_rail_hovers(window):_redraw_rail_hover(window)


    def hover_cursor(context,node,box,point):
        """Resolve cursor and transient rail readout without changing selection."""
        warp = _application.component('ui.warp')
        actions=[]
        try:return _hover_cursor(context,node,box,point,actions)
        finally:
            editor=box.get('_warp_editor') or warp_view.editor_key(context)
            action=actions[0] if actions else None
            handle=action.get('handle') if action and action.get('vertical') else None
            tone=getattr(getattr(node,'pg_chromatic',None),'active_tone',0)
            if warp.set_rail_hover(editor,node,tone,handle):
                _redraw_rail_hover(getattr(context,'window',None))


    def _hover_cursor(context,node,box,point,actions):
        """Pure control hit-testing: never call a scope/image analysis here."""
        if any(in_rect(point,box[k]) for k in ('close','linear_tab','tone_tab','color_tab')):return 'HAND_POINT'
        if scope_retry_hit(context,node,box,point):return 'HAND_POINT'
        if in_rect(point,box['header']):return 'SCROLL_XY'
        if not nodes.editable(node):return None
        try:
            if getattr(context.window_manager,'pg_stage','LINEAR')=='LINEAR' and in_rect(point,box['vectorscope']):return 'SCROLL_XY'
            if comparison_hit(context,node,box,point):return 'HAND_POINT'
            space=nodes.grading_space(node);stage=getattr(context.window_manager,'pg_stage','LINEAR')
            if stage=='CHROMATIC':
                chromatic_diagram = _application.component('ui.color')
                action=chromatic_diagram.hit(node,space,box,point)
                actions.append(action)
                if action['kind']=='slider':return 'SCROLL_Y' if action.get('vertical') else 'SCROLL_X'
                if action['kind']=='warp_dot':return None if action.get('disabled') else 'SCROLL_XY'
                if action['kind']=='dot' and action.get('on_handle'):
                    return 'SCROLL_XY' if len(action['fields'])==2 else 'SCROLL_Y'
                if action['kind'] in {'family','driver','tone'}:return 'HAND_POINT'
            elif stage=='REMAP':
                tonal_diagram = _application.component('ui.tone')
                from ...perceptual_grade_core.model import tone as tonal_math
                if in_rect(point,box['exposure']) or in_rect(point,box['fit']):return 'SCROLL_X'
                if in_rect(point,box['peak']):return 'HAND_POINT'
                model=tonal_math.model(nodes.read_tone(node),space)
                if tonal_diagram.hit_level(point,box,model) is not None:return 'SCROLL_Y'
                if tonal_diagram.hit_level(point,box,model,'after') is not None:return 'HAND_POINT'
                if tonal_diagram.hit_curve(point,box,model,context.window_manager.pg_selected_tone) is not None:return 'SCROLL_X'
            else:
                if in_rect(point,box['exposure']):return 'SCROLL_X'
                if in_rect(point,box['balance']):return 'SCROLL_XY'
                geometry=cm.diagram_geometry(nodes.read_parameters(node),space)
                if hit_handle(point,box['chart'],geometry,7*box['scale']) is not None:return 'SCROLL_XY'
        except (ValueError,ReferenceError,RuntimeError):pass
        return None


    def rectangle(rect, color):
        x, y, w, h = rect
        triangles([(x, y), (x+w, y), (x+w, y+h), (x, y), (x+w, y+h), (x, y+h)], color)


    def circle(p, r, color, hollow=False, width=1):
        pts = [(p[0]+r*math.cos(i*math.tau/32), p[1]+r*math.sin(i*math.tau/32)) for i in range(33)]
        if hollow:
            lines([q for a, b in zip(pts, pts[1:]) for q in (a, b)], color, width)
        else:
            triangles([q for a, b in zip(pts, pts[1:]) for q in (p, a, b)], color)


    def text(message, x, y, size, color=TEXT):
        blf.size(0, int(size));blf.color(0, *color);blf.position(0, x, y, 0);blf.draw(0, display(message))


    def wrap_help(message, width, size):
        """Wrap at words or CJK glyphs using the same font as the tooltip."""
        blf.size(0, int(size))
        rows=[];row=''
        for token in re.findall(r'\S+\s*', message):
            pieces=(list(token) if any(unicodedata.east_asian_width(c) in 'WF' for c in token)
                    or blf.dimensions(0,token)[0]>width else (token,))
            for piece in pieces:
                if row and blf.dimensions(0,row+piece)[0]>width:
                    rows.append(row.rstrip());row=''
                row+=piece if row else piece.lstrip()
        if row.strip():rows.append(row.rstrip())
        return rows


    def help_layout(context,state):
        """Keep the complete bubble inside its editor, including at high DPI."""
        s=state.scale;pad=10*s;gap=5*s
        width=min(380*s,context.region.width-2*gap)
        title=tr(state.help.title);size=11*s
        def rows_at(size):
            rows=[(line,True) for line in wrap_help(title,width-2*pad,size+1*s)]
            for section in state.help.sections:
                rows.append(('',False))
                if section.label:
                    rows.extend((line,True) for line in wrap_help(tr(section.label),width-2*pad,size))
                rows.extend((line,False) for line in wrap_help(section.text(),width-2*pad,size))
            return rows
        rows=rows_at(size);line_height=size+3*s
        available=context.region.height-2*gap
        while len(rows)*line_height+2*pad>available and size>6*s:
            size-=s;rows=rows_at(size);line_height=size+3*s
        height=len(rows)*line_height+2*pad
        x=min(max(gap,state.point[0]+14*s),context.region.width-width-gap)
        y=state.point[1]-height-18*s
        if y<gap:y=state.point[1]+22*s
        y=max(gap,min(y,context.region.height-height-gap))
        return (x,y,width,height),rows,size,line_height,pad


    def draw_hover_help(context,node):
        state=gizmo_help.current(context,node)
        if state is None:return
        rect,rows,size,line_height,pad=help_layout(context,state)
        x,y,w,h=rect;s=state.scale
        rectangle((x+2*s,y-2*s,w,h),(0,0,0,.35))
        rectangle(rect,theme.PANEL)
        lines([(x,y),(x+w,y),(x+w,y),(x+w,y+h),(x+w,y+h),(x,y+h),(x,y+h),(x,y)],theme.OUTLINE,s)
        for index,(message,heading) in enumerate(rows):
            if message:text(message,x+pad,y+h-pad-(index+1)*line_height,size,TEXT if heading else MUTED)


    def arrow(a, b, color, width, head):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length < 2:
            return
        lines([a, b], color, width)
        ux, uy = dx/length, dy/length
        head = min(head, length*.4)
        left = (b[0]-head*ux+head*.45*uy, b[1]-head*uy-head*.45*ux)
        right = (b[0]-head*ux-head*.45*uy, b[1]-head*uy+head*.45*ux)
        triangles([b, left, right], color)



    def scope_consumers():
        """Scopes belong to visible editor pads, not every formerly viewed node."""
        # Undo recovery owns the next RNA scan; even window discovery must wait.
        clouds = _application.component('scopes.clouds')
        if nodes.recovery_pending():
            clouds.clear()
            return {}
        result={};raster_views={};scope_demands={}
        for wm in bpy.data.window_managers:
            if not getattr(wm,'pg_diagram_enabled',False):continue
            for window in wm.windows:
                for area in window.screen.areas:
                    if area.type!='NODE_EDITOR':continue
                    region=next((region for region in area.regions if region.type=='WINDOW'),None)
                    if region is None or region.width<1 or region.height<1:continue
                    try:
                        with bpy.context.temp_override(window=window,area=area,region=region):
                            node=nodes.active_grade(bpy.context)
                            if node is None:continue
                            stage=getattr(wm,'pg_stage','LINEAR')
                            kind='linear' if stage=='LINEAR' and nodes.is_grade(node) else 'full'
                            view=(node.pg_chromatic.active_tone,node.pg_chromatic.family=='WARP') if stage=='CHROMATIC' else ('tone',) if stage=='REMAP' and nodes.is_grade(node) else ()
                            channel=_application.component('scopes.sampling').channel(node)
                            try:capture_context=_application.component('scopes.context').source_context(node)
                            except (ValueError,RuntimeError,ReferenceError) as exc:
                                capture_context={'_scope_context_error':str(exc),
                                                 'consumer':channel[1] if isinstance(channel,tuple) else None}
                            result[kind,channel]=(node,nodes.grading_space(node),view,capture_context)
                            if stage=='CHROMATIC':
                                scope_demands.setdefault(node.as_pointer(),[]).append(view)
                                if view[1]:
                                    box=layout(bpy.context)
                                    if box is not None:
                                        raster_views.setdefault(node.as_pointer(),[]).append(clouds.capture_view(node,box))
                    except (ReferenceError,RuntimeError,ValueError):continue
        clouds.replace_views(raster_views)
        clouds.replace_demands(scope_demands)
        return result

    def layout(context):
        region = context.region
        if not region or region.type != 'WINDOW':
            return None
        scale = context.preferences.system.ui_scale
        wm = context.window_manager
        node=nodes.active_grade(context)
        comparison = _application.component('scopes.reference')
        reference_connected=node is not None and comparison.connected(node)
        if getattr(wm,'pg_stage','LINEAR')=='CHROMATIC':
            chromatic_geometry = _application.component('ui.color_geometry')
            family=node.pg_chromatic.family if node is not None else 'J'
            box=chromatic_geometry.layout(region.width,region.height,scale,wm.pg_color_width,wm.pg_diagram_x,wm.pg_diagram_y,family,reference_connected)
            if box is not None:box['_warp_editor']=warp_view.editor_key(context)
            return box
        if getattr(wm,'pg_stage','LINEAR')=='REMAP':
            return tone_layout(region.width,region.height,scale,wm.pg_tone_width,wm.pg_diagram_x,wm.pg_diagram_y,reference_connected)
        return diagram_layout(region.width, region.height, scale, wm.pg_diagram_size,
                              wm.pg_diagram_x, wm.pg_diagram_y)


    def dashed_line(a, b, color, width, dash):
        distance = math.dist(a, b)
        if distance == 0:
            return
        coords = []
        for i in range(math.ceil(distance/(2*dash))):
            start, end = i*2*dash/distance, min((i*2+1)*dash/distance, 1)
            coords.extend(tuple(a[k]+(b[k]-a[k])*t for k in (0, 1)) for t in (start, end))
        lines(coords, color, width)


    def draw_value_slider(rect,label,value,lo,hi,s,precision=0):
        """Shared stage-2/3 track styling; narrow columns wrap long labels."""
        x,y,w,h=rect;message=f'{value:.{precision}f}'
        if h>=38*s:
            wrap = _application.component('localization.i18n').wrap
            rows=wrap(label,max(10,int(w/(4.5*s))))
            for i,row in enumerate(rows[:2]):text(row,x,y+h-(8+10*i)*s,9*s)
            text(message,x+w-len(message)*5.5*s,y+13*s,9*s,theme.ORANGE)
        else:
            text(label,x,y+19*s,9*s)
            text(message,x+w-max(12,len(message)*5.5)*s,y+19*s,9*s,theme.ORANGE)
        yy=y+5*s;lines([(x,yy),(x+w,yy)],theme.OUTLINE,3*s)
        pp=(x+(value-lo)/(hi-lo)*w,yy)
        lines([(x,yy),pp],theme.BLUE,3*s);circle(pp,3*s,theme.ORANGE)


    def draw_exposure(params,box):
        s=box['scale'];x,y,w,h=box['exposure']
        text(M('control.exposure.label'),x,y+19*s,10*s)
        text(tr('unit.exposure.format',value=params.exposure),x+w-46*s,y+19*s,9*s,theme.ORANGE)
        yy=y+4*s;lines([(x,yy),(x+w,yy)],theme.OUTLINE,3*s)
        pp=(x+(params.exposure+6)/12*w,yy)
        lines([(x+w/2,yy),pp],theme.ORANGE,3*s);circle(pp,3*s,theme.ORANGE)


    def draw_gain_controls(params,box):
        s=box['scale'];draw_exposure(params,box)
        x, y, w, h = box['balance']
        rectangle(box['balance'], theme.HEADER)
        lines([(x, y), (x+w, y), (x+w, y), (x+w, y+h),
               (x+w, y+h), (x, y+h), (x, y+h), (x, y)], theme.OUTLINE, s)
        text(M('diagram.balance.label'), x+5*s, y+h-11*s, 8*s, MUTED)
        ax, ay, aw, ah = box['balance_axes']
        center = balance_point(0, 0, box)
        for end, color in (((ax+aw, center[1]), COLORS['cr'][0]),
                           ((ax, center[1]), COLORS['cr'][1]),
                           ((center[0], ay+ah), COLORS['yb'][1]),
                           ((center[0], ay), COLORS['yb'][0])):
            lines([center, end], color, s)
        text(marker('scope','R'), ax+aw+3*s, center[1]-3*s, 8*s, COLORS['cr'][0])
        text(marker('scope','C'), ax-11*s, center[1]-3*s, 8*s, COLORS['cr'][1])
        text(marker('scope','Y'), center[0]+5*s, ay+ah-5*s, 8*s, COLORS['yb'][1])
        text(marker('scope','B'), center[0]+5*s, ay-2*s, 8*s, COLORS['yb'][0])
        p = balance_point(params.cr_balance, params.yb_balance, box)
        circle(p, 4*s, TEXT)
        circle(p, 2.7*s, theme.BLUE)
        text(f'{(-params.cr_balance or 0.):+.2f}', x+4*s, y+4*s, 8*s, COLORS['cr'][0])
        text(f'{(-params.yb_balance or 0.):+.2f}', x+w-31*s, y+4*s, 8*s, COLORS['yb'][0])


    def curve(chart, points, color, width=1, dashed=False):
        coords = []
        dash_length=5.
        dash_period=2*dash_length
        path_phase=0.
        for a, b in zip(points, points[1:]):
            clipped = chart.clip(a, b)
            if not dashed:
                if clipped is None:continue
                p,q=clipped
                coords.extend((p, q))
                continue
            if not all(math.isfinite(value) for value in (*a,*b)):
                # Singularities split the visual dash phase just as they split a
                # drawable path; never let one NaN poison later segments.
                path_phase=0.
                continue
            start=chart.screen(a);end=chart.screen(b)
            segment_length=math.dist(start,end)
            if clipped is not None and segment_length>0.:
                p,q=clipped
                clipped_start=math.dist(start,p)
                clipped_length=math.dist(p,q)
                phase=(path_phase+clipped_start%dash_period)%dash_period
                cursor=-phase if phase<dash_length else dash_period-phase
                while clipped_length>0. and cursor<clipped_length-1e-9:
                    t0=max(0.,cursor)/clipped_length
                    t1=min(clipped_length,cursor+dash_length)/clipped_length
                    coords.extend((
                        tuple(p[k]+(q[k]-p[k])*t0 for k in (0,1)),
                        tuple(p[k]+(q[k]-p[k])*t1 for k in (0,1))))
                    cursor+=dash_period
            if math.isfinite(segment_length):path_phase=(path_phase+segment_length%dash_period)%dash_period
            else:path_phase=0.
        lines(coords, color, width)


    def reference_hint(node):
        if editing.is_target(node):return ''
        comparison = _application.component('scopes.reference')
        return tr('hint.reference_toggle') if comparison.connected(node) else ''


    def draw_footer(node,box):
        s=box['scale'];x,y,w,h=box['rect']
        hint=reference_hint(node)
        if hint and box.get('family')=='WARP':hint=M('hint.reference_warp')
        if hint:
            wrap = _application.component('localization.i18n').wrap
            rows=wrap(hint,max(12,int((w-34*s)/(4*s))))
            for i,row in enumerate(rows):text(row,x+17*s,y+(18+12*(len(rows)-1-i))*s,8*s,MUTED)
        text(M('hint.drag_keys'),x+17*s,y+6*s,8*s,MUTED)


    def draw_board(context, node, space, params, box):
        s = box['scale'];chart = box['chart'];x, y, w, h = box['rect']
        space = nodes.grading_space(node)
        g = cm.diagram_geometry(params, space, nodes.grades.matrix(node))
        rectangle(box['rect'], theme.PANEL)
        rectangle(box['header'], theme.HEADER)
        rectangle((chart.left, chart.bottom, chart.size, chart.size), theme.PLOT)
        draw_header(box,'LINEAR',node)
        text(M('diagram.xy.title'), chart.left, chart.bottom+chart.size+27*s, 10*s, MUTED)
        grid = []
        for i in range(10):
            t = i*.1
            a, b = chart.screen((t, 0)), chart.screen((t, .9))
            c, d = chart.screen((0, t)), chart.screen((.9, t))
            grid.extend((a, b, c, d))
            text(f'{t:.1f}', a[0]-8*s, a[1]-18*s, 10*s, MUTED)
            text(f'{t:.1f}', c[0]-30*s, c[1]-4*s, 10*s, MUTED)
        lines(grid, theme.GRID, .7*s)
        locus.draw(chart)
        text(M('symbol.x'), chart.left+chart.size+7*s, chart.bottom-4*s, 11*s)
        text(M('symbol.y'), chart.left-4*s, chart.bottom+chart.size+4*s, 11*s)
        curve(chart, g['reference']+[g['reference'][0]], theme.REFERENCE, 1.3*s, dashed=True)
        curve(chart, g['graded'], theme.GRADED, 1.1*s)
        center = chart.screen(g['destination'])
        for axis in ('cr', 'yb'):
            for i, endpoint in enumerate(g['endpoints'][axis]):
                p = chart.screen(endpoint);color = COLORS[axis][i]
                arrow(center, p, color, 1.6*s, 5*s)
                circle(p, 3*s, color);circle(p, 4.5*s, color, True, .8*s)
                label = ('R', 'C') if axis == 'cr' else ('B', 'Y')
                text(marker('scope',label[i]), p[0]+6*s, p[1]+5*s, 10*s, color)
        source = chart.screen(g['source'])
        amber = theme.ORANGE
        arrow(source, center, amber, 1.6*s, 7*s)
        circle(source, 4.5*s, amber, True, 1.5*s)
        circle(source, 1.7*s, amber)
        lines([(center[0]-4*s, center[1]), (center[0]+4*s, center[1]),
               (center[0], center[1]-4*s), (center[0], center[1]+4*s)], TEXT, 1.4*s)
        text(marker('white','S'), source[0]-16*s, source[1]+9*s, 10*s, amber)
        text(marker('white','D'), center[0]+7*s, center[1]-15*s, 10*s)
        ix,iy,iw,ih=box['info'];top=iy+ih
        scopes = _application.component('scopes.sampling')
        chromatic_diagram = _application.component('ui.color')
        comparison = _application.component('scopes.reference')
        base=current=external=None
        try:
            base=scopes.get_linear(node,space,ui=True)
            current=base['linear_vectorscope'] if base is not None else None
            signal=comparison.get(node,base)
            if signal:external=signal['vectorscope']
        except (ValueError,RuntimeError,ReferenceError,TypeError):pass
        chromatic_diagram.draw_vectorscope(box['vectorscope'],current,external,s,
            M('scope.stack.output') if editing.is_target(node) else M('scope.vectorscope.draggable'))
        chromatic_diagram.draw_scope_status(node,box['vectorscope'],s,
            'full' if editing.is_target(node) else 'linear')
        draw_gain_controls(params,box)
        for i,(label,color) in enumerate(((M('symbol.rec709_dotted'),theme.REFERENCE),(M('diagram.corrected.legend'),theme.GRADED))):
            yy=top-(150+19*i)*s;a,b=(ix,yy),(ix+14*s,yy)
            if i==0:dashed_line(a,b,color,1.3*s,3*s)
            else:lines([a,b],color,1.6*s)
            text(label,ix+18*s,yy-3*s,9*s,color)
        hint=reference_hint(node)
        footer_bottom=y+12*s
        if hint:
            wrap = _application.component('localization.i18n').wrap
            rows=wrap(hint,max(12,int(iw/(4*s))))+[M('hint.drag'),M('hint.cancel'),M('hint.reset_handle')]
            for i,row in enumerate(rows):text(row,ix,footer_bottom+10*(len(rows)-1-i)*s,8*s,MUTED)
        else:
            text(M('hint.drag'),ix,footer_bottom+32*s,9*s,MUTED)
            text(M('hint.cancel'),ix,footer_bottom+16*s,9*s,MUTED)
            text(M('hint.reset_handle'),ix,footer_bottom,9*s,MUTED)


    def draw_header(box,stage,node=None):
        s=box['scale'];x,y,w,h=box['rect']
        title=M('product.wordmark')
        if editing.is_target(node):
            title=tr('stack.editing',grade=node.pg_grade.name,direction=tr('direction.'+node.entry.direction.lower()))
            limit=max(16,int((w/s-48)/6))
            if len(title)>limit:title=title[:limit-1]+'…'
        text(title,x+12*s,y+h-22*s,10*s)
        for key,title,mode in (('linear_tab',M('stage.linear.label'),'LINEAR'),('tone_tab',M('stage.tone.label'),'REMAP'),('color_tab',M('stage.color.label'),'CHROMATIC')):
            rect=box[key]
            if mode==stage:rectangle(rect,theme.BLUE)
            text(title,rect[0]+4*s,rect[1]+6*s,9*s,TEXT if mode==stage else MUTED)
        text(M('symbol.close'),x+w-23*s,y+h-23*s,17*s,MUTED)


    def draw(context=None):
        nonlocal _last_error
        context = context or bpy.context
        if not getattr(context.window_manager, 'pg_diagram_enabled', False):
            return
        node = nodes.active_grade(context)
        if node is None:
            return
        box = layout(context)
        if box is None:
            return
        blend = gpu.state.blend_get();depth = gpu.state.depth_test_get()
        try:
            gpu.state.blend_set('ALPHA');gpu.state.depth_test_set('NONE')
            # Use region-pixel coordinates and restore both matrix stacks.
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(Matrix(((2/context.region.width, 0, 0, -1),
                                                          (0, 2/context.region.height, 0, -1),
                                                          (0, 0, -1, 0), (0, 0, 0, 1))))
                if getattr(context.window_manager,'pg_stage','LINEAR')=='CHROMATIC':
                    chromatic_diagram = _application.component('ui.color')
                    chromatic_diagram.draw_board(context,node,nodes.grading_space(node),nodes.read_parameters(node),box)
                elif getattr(context.window_manager,'pg_stage','LINEAR')=='REMAP':
                    tonal_diagram = _application.component('ui.tone')
                    tonal_diagram.draw_board(context,node,nodes.grading_space(node),nodes.read_parameters(node),box)
                else:draw_board(context, node, nodes.grading_space(node), nodes.read_parameters(node), box)
                draw_hover_help(context,node)
            _last_error = ''
        except Exception as exc:
            if diagnostic_text(exc) != _last_error:
                log_exception(LOG,'Perceptual Grade diagram failed')
            _last_error = diagnostic_text(exc)
            x, y, w, h = box['rect']
            rectangle(box['rect'], theme.PANEL)
            text(tr('error.diagram_unavailable',error=error_text(exc)[:70]), x+12, y+h-30, 12, (1, .6, .3, 1))
        finally:
            gpu.state.blend_set(blend);gpu.state.depth_test_set(depth)


    class PG_OT_diagram_drag(bpy.types.Operator):
        _context = OwnedField('diagram.context')
        _node = OwnedField('diagram.node')
        _grade_start = OwnedField('diagram.grade_start')
        _area = OwnedField('diagram.area')
        _wm = OwnedField('diagram.wm')
        _window = OwnedField('diagram.window')
        _drag_timer = OwnedField('diagram.drag_timer')
        bl_idname = 'perceptual_grade.diagram_drag'
        bl_label = native('action.drag.label')
        bl_translation_context = CONTEXT
        bl_options = {'UNDO', 'BLOCKING'}

        @classmethod
        def poll(cls, context):
            return (context.area and context.area.type == 'NODE_EDITOR' and
                    getattr(context.window_manager, 'pg_diagram_enabled', False) and nodes.active_grade(context) is not None)

        def invoke(self, context, event):
            try:
                return PG_OT_diagram_drag._invoke(self, context, event)
            except BaseException:
                finish_drag(self)
                raise

        def _invoke(self, context, event):
            self._box = layout(context)
            if self._box is None:
                return {'PASS_THROUGH'}
            point = (event.mouse_region_x, event.mouse_region_y)
            if not in_rect(point, self._box['rect']):
                return {'PASS_THROUGH'}
            gizmo_help.clear(context.window)
            scheduler = _application.component('graph.scheduler')
            self._session = scheduler.session()
            self._lifecycle_closed = False
            self._context = context
            self._node = nodes.active_grade(context)
            self._grade_start = nodes.grades.selected(self._node)
            self._area = context.area
            self._mode = 'consume'
            self._wm = context.window_manager
            self._window = context.window
            self._last_mouse = (event.mouse_x, event.mouse_y)
            self._press_mouse=self._last_mouse
            self._scope_dragged=False;self._scope_acted=False;self._white_drag=None
            self._start_pos = (self._wm.pg_diagram_x, self._wm.pg_diagram_y)
            self._space = None
            self._start = None
            self._stage=getattr(self._wm,'pg_stage','LINEAR')
            # Stack targets expose the same settings, but a Stage-1 gesture does
            # not need either model. Avoid touching their RNA focus/settings until
            # the user is actually editing that page.
            self._chromatic_start=None;self._focus_start=None;self._tone_start=None
            if self._stage=='CHROMATIC':
                self._chromatic_start=nodes.read_chromatic(self._node)
                ui=self._node.pg_chromatic
                self._focus_start=(ui.family,ui.active_tone,ui.active_chroma,ui.active_handle)
            elif self._stage=='REMAP':
                self._tone_start=nodes.read_tone(self._node)
            self._selected_tone_start=getattr(self._wm,'pg_selected_tone',3)
            self._tone_values=None
            self._pending_color=None;self._drag_timer=None;self._last_commit=0.
            self._schedule_gesture=None
            self._warp_fit_token=None;self._warp_scale_token=None;self._warp_dot_token=None
            try:
                self._space = nodes.grading_space(self._node)
                self._scene_space = nodes.working_space()
                self._linear_start = nodes.grades.matrix(self._node)
                self._start = nodes.read_parameters(self._node)
            except ValueError as exc:
                _application.component('integration.outcomes').failure(self._node, 'diagram', exc)
            # Consume an entire click, including its release, even on empty/read-
            # only/error parts of the board. No node behind the board receives it.
            if event.type != 'LEFTMOUSE':
                _active_drags[id(self)] = retain(self,'diagram.active_operator')
                context.window_manager.modal_handler_add(self)
                return {'RUNNING_MODAL'}
            retry_kind=scope_retry_hit(context,self._node,self._box,point)
            if retry_kind is not None:
                if retry_kind in {'gpu_details','gpu_retry'}:
                    color = _application.component('ui.color')
                    try:color.gpu_action(context,retry_kind,role=scope_gpu_role(context))
                    except RuntimeError as exc:_application.component('integration.outcomes').failure(self._node, 'diagram', exc)
                else:
                    jobs = _application.component('scopes.jobs')
                    jobs.retry(self._node,retry_kind)
                self._area.tag_redraw()
            elif in_rect(point, self._box['close']):
                self._mode = 'close'
            elif in_rect(point,self._box['linear_tab']):self._mode='stage_linear'
            elif in_rect(point,self._box['tone_tab']):self._mode='stage_tone'
            elif in_rect(point,self._box['color_tab']):self._mode='stage_color'
            elif in_rect(point, self._box['header']):
                self._mode = 'move'
            elif self._start is not None and nodes.editable(self._node) and self._stage=='LINEAR' and in_rect(point,self._box['vectorscope']):
                scheduler = _application.component('graph.scheduler')
                self._schedule_gesture=scheduler.begin(self._node)
                chromatic_geometry = _application.component('ui.color_geometry')
                self._mode='scope_white'
                self._white_scope_rect=chromatic_geometry.square(self._box['vectorscope'])
                # Defer the sparse confidence fit until movement crosses the threshold.
            elif self._start is not None and nodes.editable(self._node) and comparison_hit(context,self._node,self._box,point):
                self._mode='comparison'
            elif self._start is not None and nodes.editable(self._node):
                scheduler = _application.component('graph.scheduler')
                self._schedule_gesture=scheduler.begin(self._node)
                if self._stage=='CHROMATIC':
                    chromatic_diagram = _application.component('ui.color')
                    self._mode='chromatic'
                    self._color_action=chromatic_diagram.hit(self._node,self._space,self._box,point)
                    self._color_accumulator={n:getattr(self._chromatic_start,n) for n in self._color_action.get('fields',())}
                    if (self._node.pg_chromatic.family=='WARP' and
                            self._color_action['kind'] in {'warp_dot','slider'}):
                        # Sliders may write on mouse-down, so freeze before apply/
                        # absolute_fit rather than waiting for the first movement.
                        plot=self._color_action.get('plot')
                        if plot is None:
                            plot=chromatic_diagram.control_state(self._node,self._space,self._box)[3]
                        self._warp_fit_token=warp_view.begin(self._box.get('_warp_editor'),
                            self._node,self._node.pg_chromatic.active_tone,plot.xmax)
                        if self._color_action.get('vertical'):
                            warp = _application.component('ui.warp')
                            self._color_action['scale_model']=nodes.color_model(self._node,self._space)
                            self._warp_scale_token=warp.begin_scale_drag(self._box.get('_warp_editor'),
                                self._node,self._node.pg_chromatic.active_tone,self._color_action['handle'],self._color_action['rect'])
                    try:
                        if self._color_action['kind']=='warp_dot':
                            warp = _application.component('ui.warp')
                            reset = _application.component('ui.warp_drag').reset
                            reset(nodes.color_model(self._node,self._space),self._color_action)
                            self._warp_dot_token=warp.begin_drag(self._box.get('_warp_editor'),self._node,self._color_action)
                        chromatic_diagram.apply_click(self._node,self._color_action)
                        if self._color_action['kind'] in {'dot','slider','warp_dot'}:
                            if hasattr(self._wm,'event_timer_add'):
                                self._drag_timer=self._wm.event_timer_add(1/60,window=context.window)
                                _drag_timers[id(self)]=(retain(self._wm,'diagram.timer_wm'),retain(self._drag_timer,'diagram.timer'))
                        self.absolute_fit(point[0])
                    except (ReferenceError,ValueError,RuntimeError):
                        end_color_drag(self,commit=False)
                        raise
                    self._area.tag_redraw()
                elif self._stage=='REMAP':
                    tonal_diagram = _application.component('ui.tone')
                    from ...perceptual_grade_core.model import tone as tm
                    self._tone_model=tm.model(self._tone_start,self._space)
                    if in_rect(point,self._box['exposure']):self._mode='exposure'
                    elif in_rect(point,self._box['peak']):self._mode='peak'
                    elif in_rect(point,self._box['fit']):
                        self._mode='fit';self.absolute_fit(point[0])
                    else:
                        index=None if editing.is_target(self._node) else tonal_diagram.hit_level(point,self._box,self._tone_model)
                        if index is not None:
                            self._mode='tone_level';self._level=index
                            self._wm.pg_selected_tone=index
                            self._level_j=float(self._tone_model.source[index])
                        else:
                            index=None if editing.is_target(self._node) else tonal_diagram.hit_level(point,self._box,self._tone_model,'after')
                            if index is not None:
                                self._mode='tone_select';self._wm.pg_selected_tone=index
                            else:
                                index=tonal_diagram.hit_curve(point,self._box,self._tone_model,self._wm.pg_selected_tone)
                                if index is not None:
                                    self._mode='tone_curve';self._level=index;self._wm.pg_selected_tone=index
                                    self._level_j=float(self._tone_model.source[index])
                    self._area.tag_redraw()
                elif in_rect(point, self._box['exposure']):
                    self._mode = 'exposure'
                elif in_rect(point, self._box['balance']):
                    self._mode = 'balance'
                else:
                    geometry = cm.diagram_geometry(self._start, self._space)
                    selected = hit_handle(point, self._box['chart'], geometry, 7*self._box['scale'])
                    if selected is not None:
                        self._mode, self._sign = selected
                        self._logical = geometry['source'] if self._mode == 'white' else geometry['endpoints'][self._mode][0 if self._sign == 1 else 1]
            _active_drags[id(self)] = retain(self,'diagram.active_operator')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}

        def restore(self):
            set_hover_cursor(self._window)
            end_color_drag(self,commit=False)
            if self._mode == 'move':
                self._wm.pg_diagram_x, self._wm.pg_diagram_y = self._start_pos
            elif self._mode=='chromatic':
                from ...perceptual_grade_core.model import color as ch
                nodes.write_chromatic(self._node,{n:getattr(self._chromatic_start,n) for n in ch.PARAM_NAMES})
                ui=self._node.pg_chromatic;ui.family,ui.active_tone,ui.active_chroma,ui.active_handle=self._focus_start
            elif self._mode in {'tone_level','tone_curve','fit'}:
                from ...perceptual_grade_core.model import tone as tm
                nodes.write_tone(self._node,{n:getattr(self._tone_start,n) for n in tm.PARAM_NAMES})
                self._wm.pg_selected_tone=self._selected_tone_start
            elif self._mode=='tone_select':self._wm.pg_selected_tone=self._selected_tone_start
            elif self._mode not in {'consume', 'comparison','close','stage_linear','stage_tone','stage_color','peak','tone_select'}:
                nodes.grades.write_matrix(self._node, self._linear_start)
                nodes.grades.publish(nodes.grades.data_owner(self._node))

            if getattr(self, '_grade_start', None) is None and nodes.grades.selected(self._node) is not None:
                grade = self._node.pg_grade
                nodes.grades.unlink(self._node)
                if grade.users == int(grade.use_fake_user):
                    remove_node_group(grade)

        def absolute_fit(self,mouse_x):
            """Discrete tracks follow the pointer: integers and half-saturation steps."""
            if self._mode=='fit':
                x,_,w,_=self._box['fit_track']
                value=round(-2+8*max(0,min(1,(mouse_x-x)/w)))
                nodes.write_tone(self._node,dict(fit=value))
                return True
            if self._mode=='chromatic' and self._color_action['kind']=='slider':
                controls = _application.component('model.controls')
                action=self._color_action;field=action['fields'][0];meta=controls.meta(field)
                if controls.step(field) is None:return False
                x,_,w,_=action['rect'];lo,hi=meta[3:5]
                value=controls.snap(field,lo+(hi-lo)*max(0,min(1,(mouse_x-x)/w)))
                self._color_accumulator[field]=value;self._pending_color={field:value}
                if self._drag_timer is None or time.monotonic()-self._last_commit>=1/60:flush_color(self)
                return True
            return False

        def reset_dragged(self):
            """Reset only this handle; Esc still restores the pre-gesture snapshot."""
            mode=self._mode
            if mode=='move':self._wm.pg_diagram_x=24.;self._wm.pg_diagram_y=24.
            elif mode=='exposure':nodes.write_values(self._node,dict(exposure=0.))
            elif mode=='balance':nodes.write_values(self._node,dict(cr_balance=0.,yb_balance=0.))
            elif mode in {'white','scope_white','cr','yb'}:
                if mode in {'white','scope_white'}:
                    wp=cm.white(self._space)
                    nodes.write_values(self._node,dict(source_x=wp[0],source_y=wp[1]))
                else:nodes.write_values(self._node,{mode+'_spread':0.,mode+'_heading':0.})
                g=cm.diagram_geometry(nodes.read_parameters(self._node),self._space)
                if mode=='scope_white':
                    self._scope_acted=True;self._press_mouse=self._last_mouse
                    if self._white_drag is not None:self._white_drag.reset(nodes.read_parameters(self._node))
                else:self._logical=g['source'] if mode=='white' else g['endpoints'][mode][0 if self._sign==1 else 1]
            elif mode in {'tone_level','tone_curve'}:
                from ...perceptual_grade_core.model import tone as tm
                current=nodes.read_tone(self._node);model=tm.model(current,self._space)
                value=tm.clamp_landmark(current,self._level,float(model.target_nits[self._level]),self._space)
                nodes.write_tone(self._node,{tm.LEVEL_FIELDS[self._level]:value})
                self._level_j=float(model.nits_to_j(value))
            elif mode=='fit':
                from ...perceptual_grade_core.model import tone as tm
                nodes.write_tone(self._node,dict(fit=tm.Parameters().fit))
            elif mode=='chromatic' and self._color_action['kind'] in {'dot','slider','warp_dot'}:
                from ...perceptual_grade_core.model import color as ch
                action=self._color_action
                if action['kind']=='slider' and action['fields'][0].startswith('web_scale_'):
                    warp_scale = _application.component('ui.warp_scale')
                    from dataclasses import replace
                    params=nodes.read_chromatic(self._node)
                    if self._pending_color:params=replace(params,**self._pending_color)
                    model=ch.model(nodes.read_tone(self._node),params,self._space)
                    _,_,row,handle=action['fields'][0].split('_')
                    values=warp_scale.edit(model,int(row),int(handle),0.)
                    if not values:return
                    action['scale_model']=ch.model(nodes.read_tone(self._node),replace(params,**values),self._space)
                    self._color_accumulator[action['fields'][0]]=0.
                else:
                    defaults=ch.Parameters();values={name:getattr(defaults,name) for name in action['fields']}
                self._color_accumulator.update(values);self._pending_color=values;flush_color(self)
                if self._color_action['kind']=='warp_dot':
                    model=nodes.color_model(self._node,self._space)
                    reset = _application.component('ui.warp_drag').reset
                    reset(model,self._color_action)

        def modal(self, context, event):
            result = None
            try:
                result = PG_OT_diagram_drag._modal(self, context, event)
                return result
            finally:
                if result != {'RUNNING_MODAL'}:
                    finish_drag(self)

        def _modal(self, context, event):
            scheduler = _application.component('graph.scheduler')
            if (getattr(self, '_lifecycle_closed', False) or
                    getattr(self, '_session', scheduler.session()) != scheduler.session()):
                _active_drags.pop(id(self), None)
                return {'CANCELLED'}
            try:
                if editing.is_target(self._node) and not editing.valid(self._node):
                    end_color_drag(self,commit=False)
                    set_hover_cursor(self._window)
                    return {'CANCELLED'}
                if (not _registered or not (nodes.is_grade(self._node) or editing.is_target(self._node)) or
                        not self._wm.pg_diagram_enabled or
                        (self._space is not None and (nodes.grading_space(self._node) != self._space or nodes.working_space() != self._scene_space))):
                    self.restore()
                    return {'CANCELLED'}
                if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                    self.restore();self._area.tag_redraw()
                    return {'CANCELLED'}
                if event.type=='R' and event.value=='PRESS':
                    self.reset_dragged();self._area.tag_redraw()
                    return {'RUNNING_MODAL'}
                if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'} and event.value == 'RELEASE':
                    set_hover_cursor(self._window)
                    end_color_drag(self)
                    self._area.tag_redraw()
                    if self._mode == 'close':
                        self._wm.pg_diagram_enabled = False
                        editing.close(context)
                        self._area.tag_redraw()
                    if self._mode in {'stage_linear','stage_tone','stage_color'}:
                        # Header entry also works with a synthetic WindowManager in
                        # tests and avoids relying only on an RNA update callback.
                        if self._mode=='stage_color' and self._stage!='CHROMATIC':
                            ui = _application.component('ui.panel')
                            ui.land_on_lightness(context)
                        self._wm.pg_stage={'stage_linear':'LINEAR','stage_tone':'REMAP','stage_color':'CHROMATIC'}[self._mode]
                        self._area.tag_redraw()
                    elif self._mode=='peak':bpy.ops.wm.call_menu(name='PG_MT_peak')
                    elif self._mode=='scope_white' and event.type=='LEFTMOUSE' and not self._scope_dragged and not self._scope_acted:
                        comparison = _application.component('scopes.reference')
                        if comparison.connected(self._node) and in_rect((event.mouse_region_x,event.mouse_region_y),self._white_scope_rect):
                            comparison.toggle(self._node)
                    elif self._mode=='comparison':
                        comparison = _application.component('scopes.reference')
                        comparison.toggle(self._node)
                    keyframe_drag(self, context)
                    return {'FINISHED'}
                if event.type=='TIMER' and getattr(event,'timer',None)==getattr(self,'_drag_timer',None):
                    flush_color(self);self._area.tag_redraw()
                if event.type == 'MOUSEMOVE':
                    dx, dy = event.mouse_x-self._last_mouse[0], event.mouse_y-self._last_mouse[1]
                    self._last_mouse = (event.mouse_x, event.mouse_y)
                    if self._mode == 'move':
                        self._wm.pg_diagram_x += dx/self._box['scale']
                        self._wm.pg_diagram_y += dy/self._box['scale']
                    elif self._mode=='chromatic':
                        chromatic_geometry = _application.component('ui.color_geometry')
                        action=self._color_action
                        if action['kind']=='warp_dot':
                            warp = _application.component('ui.warp')
                            # Use the latest complete accepted edit when the 60 Hz publication is pending.
                            from dataclasses import replace
                            from ...perceptual_grade_core.model import color as ch
                            params=nodes.read_chromatic(self._node)
                            if self._pending_color:params=replace(params,**self._pending_color)
                            model=ch.model(nodes.read_tone(self._node),params,self._space)
                            values=warp.drag(model,action,dx,dy)
                            if values:
                                self._pending_color=values
                                if self._drag_timer is None or time.monotonic()-self._last_commit>=1/60:flush_color(self)
                        elif action['kind'] in {'dot','slider'} and not self.absolute_fit(event.mouse_region_x):
                            values=chromatic_geometry.drag_values(nodes.read_chromatic(self._node),action['fields'],dx,dy,action.get('plot'),action.get('rect'),action['kind']=='slider',self._color_accumulator)
                            if action['kind']=='slider' and action['fields'][0].startswith('web_scale_'):
                                warp_scale = _application.component('ui.warp_scale')
                                from ...perceptual_grade_core.model import color as ch
                                baseline=action.get('scale_model')
                                if baseline is None:
                                    baseline=ch.model(nodes.read_tone(self._node),self._chromatic_start,self._space)
                                    action['scale_model']=baseline
                                _,_,row,handle=action['fields'][0].split('_')
                                values=warp_scale.edit(baseline,int(row),int(handle),values[action['fields'][0]])
                            if values:
                                self._pending_color=values
                                if self._drag_timer is None or time.monotonic()-self._last_commit>=1/60:flush_color(self)
                    elif self._mode=='scope_white':
                        if not self._scope_dragged:
                            dx,dy=event.mouse_x-self._press_mouse[0],event.mouse_y-self._press_mouse[1]
                            if math.hypot(dx,dy)<3*self._box['scale']:return {'RUNNING_MODAL'}
                            white_drag = _application.component('ui.white_drag')
                            self._scope_dragged=True
                            self._white_drag=white_drag.from_node(self._node,self._space,nodes.read_parameters(self._node))
                        if not getattr(self._white_drag,'interactive',True):
                            # A Stage-1-only correction cannot predict a whole
                            # Stack through nonlinear stages or inverse solving.
                            # Keep this gesture inert until an exact evaluator is
                            # available, and remove any stale transient prediction.
                            self._area.tag_redraw()
                            return {'RUNNING_MODAL'}
                        chart=self._box['chart'];xy_per_pixel=(chart.high-chart.low)/chart.size
                        values=self._white_drag.move_pixels(nodes.read_parameters(self._node),(dx,dy),xy_per_pixel)
                        nodes.write_values(self._node,values)
                    elif self._mode in {'exposure', 'balance'}:
                        nodes.write_values(self._node, gain_drag(self._mode, nodes.read_parameters(self._node), dx, dy, self._box))
                    elif self._mode in {'tone_level','tone_curve'}:
                        from ...perceptual_grade_core.model import tone as tm
                        delta=dx/self._box['chart'].rect[2] if self._mode=='tone_curve' else dy/self._box['before'][3]
                        self._level_j+=delta*self._tone_model.plot_max*FINE_GAIN
                        current=nodes.read_tone(self._node);field=tm.LEVEL_FIELDS[self._level]
                        value=float(self._tone_model.j_to_nits(max(0,self._level_j)))
                        value=tm.clamp_landmark(current,self._level,value,self._space)
                        nodes.write_tone(self._node,{field:value})
                        self._level_j=float(self._tone_model.nits_to_j(value))
                    elif self._mode=='fit':
                        self.absolute_fit(event.mouse_region_x)
                    elif self._mode in {'white','cr','yb'}:
                        chart = self._box['chart'];k = (chart.high-chart.low)/chart.size*FINE_GAIN
                        self._logical = (self._logical[0]+dx*k, self._logical[1]+dy*k)
                        p = nodes.read_parameters(self._node)
                        values = cm.drag_white(self._logical) if self._mode == 'white' else cm.drag_axis(self._logical, self._mode, self._sign, p, self._space)
                        nodes.write_values(self._node, values)
                        updated = cm.diagram_geometry(nodes.read_parameters(self._node), self._space)
                        self._logical = updated['source'] if self._mode == 'white' else updated['endpoints'][self._mode][0 if self._sign == 1 else 1]
                    self._area.tag_redraw()
                return {'RUNNING_MODAL'}
            except (ReferenceError, ValueError, RuntimeError):
                log_exception(LOG,'Perceptual Grade drag interrupted')
                try:
                    self.restore()
                except (ReferenceError, ValueError, RuntimeError):
                    pass
                set_hover_cursor(self._window)
                return {'CANCELLED'}


    def pointer_region(window, event):
        """Locate a canvas under a window-coordinate pointer, excluding sidebars."""
        point = (event.mouse_x, event.mouse_y)
        for area in window.screen.areas:
            if area.type != 'NODE_EDITOR' or not in_rect(point, (area.x, area.y, area.width, area.height)):
                continue
            regions = list(area.regions)
            if any(r.type != 'WINDOW' and r.width > 1 and r.height > 1 and
                   in_rect(point, (r.x, r.y, r.width, r.height)) for r in regions):
                return None
            for region in regions:
                if region.type == 'WINDOW' and in_rect(point, (region.x, region.y, region.width, region.height)):
                    return area, region
        return None


    def pointer_event(event):
        return ('MOUSE' in event.type or event.type.startswith(('WHEEL', 'TRACKPAD', 'NDOF_')))


    class PG_OT_diagram_events(bpy.types.Operator):
        """Window-modal router: precedes node buttons, selection and editor tools.

        It passes through outside the actual drawn panel and never captures an
        outside-origin drag halfway through. A separate undoable operator owns
        each control drag. The runtime token invalidates old routers on reload.
        """
        bl_idname = 'perceptual_grade.diagram_events'
        bl_label = 'Perceptual Grade Diagram Events'
        bl_translation_context = CONTEXT
        bl_options = {'INTERNAL'}

        def invoke(self, context, event):
            self._key = context.window.as_pointer()
            if self._key in _routers:
                return {'CANCELLED'}
            self._token = object()
            _routers[self._key] = self._token
            self._outside_buttons = set()
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}

        def modal(self, context, event):
            if not _registered or _routers.get(self._key) is not self._token:
                if _routers.get(self._key) is self._token:
                    _routers.pop(self._key, None)
                    set_hover_cursor(context.window)
                    gizmo_help.clear(context.window)
                    clear_rail_hover(context.window)
                return {'CANCELLED', 'PASS_THROUGH'}
            _application.component('scopes.context').remember(context,event)
            warp_edit = _application.component('ui.warp_edit')
            warp_edit.observe_event(context, event)
            if warp_edit.active(context.window):
                # Native number/text buttons own the complete sidebar gesture,
                # including when its pointer travels over the overlay.
                return {'PASS_THROUGH'}
            if not getattr(context.window_manager, 'pg_diagram_enabled', False):
                set_hover_cursor(context.window);gizmo_help.clear(context.window)
                return {'PASS_THROUGH'}
            if event.type=='WINDOW_DEACTIVATE':
                set_hover_cursor(context.window);gizmo_help.clear(context.window)
                clear_rail_hover(context.window)
            if not pointer_event(event):
                if event.value=='PRESS':gizmo_help.clear(context.window)
                return {'PASS_THROUGH'}
            if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'} and event.value == 'PRESS':
                # A native modal tool may have consumed its release. A fresh press
                # begins a new gesture; never leave the panel stuck in pass-through.
                self._outside_buttons.clear()
            # Once a gesture starts outside the overlay, native tools own it to
            # release, even if the pointer crosses the panel along the way.
            if self._outside_buttons:
                set_hover_cursor(context.window)
                gizmo_help.clear(context.window)
                clear_rail_hover(context.window)
                if event.value == 'RELEASE':
                    self._outside_buttons.discard(event.type)
                return {'PASS_THROUGH'}
            target = pointer_region(context.window, event)
            if target:
                area, region = target
                with context.temp_override(area=area, region=region):
                    box = layout(context)
                    point = (event.mouse_x-region.x, event.mouse_y-region.y)
                    if box and nodes.active_grade(context) is not None and in_rect(point, box['rect']):
                        node=nodes.active_grade(context)
                        set_hover_cursor(context.window,hover_cursor(context,node,box,point))
                        if event.type=='MOUSEMOVE' and gizmo_help.enabled(context):
                            try:
                                help=gizmo_help.at(context,node,box,point)
                                gizmo_help.update(context.window,region,gizmo_help.identity(context,node),point,box['scale'],help)
                            except (ValueError,ReferenceError,RuntimeError):gizmo_help.clear(context.window)
                        else:gizmo_help.clear(context.window)
                        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'} and event.value == 'PRESS':
                            bpy.ops.perceptual_grade.diagram_drag('INVOKE_DEFAULT')
                        return {'RUNNING_MODAL'}
            set_hover_cursor(context.window)
            gizmo_help.clear(context.window)
            clear_rail_hover(context.window)
            if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'} and event.value == 'PRESS':
                self._outside_buttons.add(event.type)
            return {'PASS_THROUGH'}


    def ensure_routers():
        if nodes.recovery_pending() or not _registered or bpy.app.background:
            return
        context = bpy.context
        if not getattr(context.window_manager, 'pg_diagram_enabled', False):
            cleanup_hover_cursors()
        live = {window.as_pointer() for window in context.window_manager.windows}
        live_editors={(window.as_pointer(),area.as_pointer(),region.as_pointer())
            for window in context.window_manager.windows for area in window.screen.areas
            if area.type=='NODE_EDITOR' for region in area.regions if region.type=='WINDOW'}
        warp_view.prune(live_editors,lambda node:editing.valid(node) if editing.is_target(node) else nodes.is_grade(node))
        warp = _application.component('ui.warp')
        warp.prune_rails(live_editors,lambda node:editing.valid(node) if editing.is_target(node) else nodes.is_grade(node))
        gizmo_help.prune(live)
        if gizmo_help.enabled(context):gizmo_help.poll()
        else:gizmo_help.clear()
        for key in set(_routers)-live:
            del _routers[key]
            _hover_cursors.pop(key,None)
        for window in context.window_manager.windows:
            if window.as_pointer() not in _routers:
                with context.temp_override(window=window):
                    bpy.ops.perceptual_grade.diagram_events('INVOKE_DEFAULT')


    draw = _application.component('integration.outcomes').callback(draw, control='diagram')

    def register():
        nonlocal _registered, _handle
        try:
            for cls in (PG_OT_diagram_drag, PG_OT_diagram_events):
                _application.register_class(cls)
                _classes.append(cls)
            if not bpy.app.background:
                _handle = bpy.types.SpaceNodeEditor.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')
            _registered = True
        except Exception:
            unregister()
            raise


    def unregister():
        pad_background = _application.component('ui.pad_background')
        warp_samples = _application.component('ui.warp_samples')
        pad_background.clear()
        warp_samples.clear()
        cleanup_drag_timers()
        cleanup_hover_cursors()
        nonlocal _handle, _registered
        _registered = False
        _routers.clear()
        if _handle is not None:
            bpy.types.SpaceNodeEditor.draw_handler_remove(_handle, 'WINDOW')
            _handle = None
        _shaders.clear()
        for cls in reversed(_classes):
            _application.unregister_class(cls)
        _classes.clear()

    def _set_export_COLORS(value):
        nonlocal COLORS
        COLORS = value

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export_MUTED(value):
        nonlocal MUTED
        MUTED = value

    def _set_export_TEXT(value):
        nonlocal TEXT
        TEXT = value

    def _set_export__active_drags(value):
        nonlocal _active_drags
        _active_drags = value

    def _set_export__classes(value):
        nonlocal _classes
        _classes = value

    def _set_export__drag_timers(value):
        nonlocal _drag_timers
        _drag_timers = value

    def _set_export__handle(value):
        nonlocal _handle
        _handle = value

    def _set_export__hover_cursors(value):
        nonlocal _hover_cursors
        _hover_cursors = value

    def _set_export__last_error(value):
        nonlocal _last_error
        _last_error = value

    def _set_export__registered(value):
        nonlocal _registered
        _registered = value

    def _set_export__routers(value):
        nonlocal _routers
        _routers = value

    def _set_export__shaders(value):
        nonlocal _shaders
        _shaders = value

    def _set_export_shader(value):
        nonlocal shader
        shader = value

    def _set_export_lines(value):
        nonlocal lines
        lines = value

    def _set_export_triangles(value):
        nonlocal triangles
        triangles = value

    def _set_export_CachedTriangles(value):
        nonlocal CachedTriangles
        CachedTriangles = value

    def _set_export_ColoredMesh(value):
        nonlocal ColoredMesh
        ColoredMesh = value

    def _set_export_colored_triangles(value):
        nonlocal colored_triangles
        colored_triangles = value

    def _set_export_flush_color(value):
        nonlocal flush_color
        flush_color = value

    def _set_export_end_color_drag(value):
        nonlocal end_color_drag
        end_color_drag = value

    def _set_export_keyframe_drag(value):
        nonlocal keyframe_drag
        keyframe_drag = value

    def _set_export_release_drag_references(value):
        nonlocal release_drag_references
        release_drag_references = value

    def _set_export_finish_drag(value):
        nonlocal finish_drag
        finish_drag = value

    def _set_export_settle_drags(value):
        nonlocal settle_drags
        settle_drags = value

    def _set_export_cleanup_drag_timers(value):
        nonlocal cleanup_drag_timers
        cleanup_drag_timers = value

    def _set_export_set_hover_cursor(value):
        nonlocal set_hover_cursor
        set_hover_cursor = value

    def _set_export_cleanup_hover_cursors(value):
        nonlocal cleanup_hover_cursors
        cleanup_hover_cursors = value

    def _set_export_scope_status_target(value):
        nonlocal scope_status_target
        scope_status_target = value

    def _set_export_scope_retry_hit(value):
        nonlocal scope_retry_hit
        scope_retry_hit = value

    def _set_export_scope_gpu_role(value):
        nonlocal scope_gpu_role
        scope_gpu_role = value

    def _set_export_comparison_hit(value):
        nonlocal comparison_hit
        comparison_hit = value

    def _set_export__redraw_rail_hover(value):
        nonlocal _redraw_rail_hover
        _redraw_rail_hover = value

    def _set_export_clear_rail_hover(value):
        nonlocal clear_rail_hover
        clear_rail_hover = value

    def _set_export_hover_cursor(value):
        nonlocal hover_cursor
        hover_cursor = value

    def _set_export__hover_cursor(value):
        nonlocal _hover_cursor
        _hover_cursor = value

    def _set_export_rectangle(value):
        nonlocal rectangle
        rectangle = value

    def _set_export_circle(value):
        nonlocal circle
        circle = value

    def _set_export_text(value):
        nonlocal text
        text = value

    def _set_export_wrap_help(value):
        nonlocal wrap_help
        wrap_help = value

    def _set_export_help_layout(value):
        nonlocal help_layout
        help_layout = value

    def _set_export_draw_hover_help(value):
        nonlocal draw_hover_help
        draw_hover_help = value

    def _set_export_arrow(value):
        nonlocal arrow
        arrow = value

    def _set_export_scope_consumers(value):
        nonlocal scope_consumers
        scope_consumers = value

    def _set_export_layout(value):
        nonlocal layout
        layout = value

    def _set_export_dashed_line(value):
        nonlocal dashed_line
        dashed_line = value

    def _set_export_draw_value_slider(value):
        nonlocal draw_value_slider
        draw_value_slider = value

    def _set_export_draw_exposure(value):
        nonlocal draw_exposure
        draw_exposure = value

    def _set_export_draw_gain_controls(value):
        nonlocal draw_gain_controls
        draw_gain_controls = value

    def _set_export_curve(value):
        nonlocal curve
        curve = value

    def _set_export_reference_hint(value):
        nonlocal reference_hint
        reference_hint = value

    def _set_export_draw_footer(value):
        nonlocal draw_footer
        draw_footer = value

    def _set_export_draw_board(value):
        nonlocal draw_board
        draw_board = value

    def _set_export_draw_header(value):
        nonlocal draw_header
        draw_header = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export_PG_OT_diagram_drag(value):
        nonlocal PG_OT_diagram_drag
        PG_OT_diagram_drag = value

    def _set_export_pointer_region(value):
        nonlocal pointer_region
        pointer_region = value

    def _set_export_pointer_event(value):
        nonlocal pointer_event
        pointer_event = value

    def _set_export_PG_OT_diagram_events(value):
        nonlocal PG_OT_diagram_events
        PG_OT_diagram_events = value

    def _set_export_ensure_routers(value):
        nonlocal ensure_routers
        ensure_routers = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component(readers={
        "__name__": lambda: __name__,
        'COLORS': lambda: COLORS,
        'CONTEXT': lambda: CONTEXT,
        'CachedTriangles': lambda: CachedTriangles,
        'ColoredMesh': lambda: ColoredMesh,
        'FINE_GAIN': lambda: FINE_GAIN,
        'LOG': lambda: LOG,
        'M': lambda: M,
        'MUTED': lambda: MUTED,
        'Matrix': lambda: Matrix,
        'OwnedField': lambda: OwnedField,
        'PG_OT_diagram_drag': lambda: PG_OT_diagram_drag,
        'PG_OT_diagram_events': lambda: PG_OT_diagram_events,
        'TEXT': lambda: TEXT,
        '_active_drags': lambda: _active_drags,
        '_classes': lambda: _classes,
        '_drag_timers': lambda: _drag_timers,
        '_handle': lambda: _handle,
        '_hover_cursor': lambda: _hover_cursor,
        '_hover_cursors': lambda: _hover_cursors,
        '_last_error': lambda: _last_error,
        '_redraw_rail_hover': lambda: _redraw_rail_hover,
        '_registered': lambda: _registered,
        '_routers': lambda: _routers,
        '_shaders': lambda: _shaders,
        'arrow': lambda: arrow,
        'balance_point': lambda: balance_point,
        'batch_for_shader': lambda: batch_for_shader,
        'blf': lambda: blf,
        'bpy': lambda: bpy,
        'circle': lambda: circle,
        'cleanup_drag_timers': lambda: cleanup_drag_timers,
        'cleanup_hover_cursors': lambda: cleanup_hover_cursors,
        'clear_rail_hover': lambda: clear_rail_hover,
        'cm': lambda: cm,
        'colored_triangles': lambda: colored_triangles,
        'comparison_hit': lambda: comparison_hit,
        'curve': lambda: curve,
        'dashed_line': lambda: dashed_line,
        'diagnostic_text': lambda: diagnostic_text,
        'diagram_layout': lambda: diagram_layout,
        'display': lambda: display,
        'draw': lambda: draw,
        'draw_board': lambda: draw_board,
        'draw_exposure': lambda: draw_exposure,
        'draw_footer': lambda: draw_footer,
        'draw_gain_controls': lambda: draw_gain_controls,
        'draw_header': lambda: draw_header,
        'draw_hover_help': lambda: draw_hover_help,
        'draw_value_slider': lambda: draw_value_slider,
        'editing': lambda: editing,
        'end_color_drag': lambda: end_color_drag,
        'ensure_routers': lambda: ensure_routers,
        'error_text': lambda: error_text,
        'exposure_point': lambda: exposure_point,
        'finish_drag': lambda: finish_drag,
        'flush_color': lambda: flush_color,
        'gain_drag': lambda: gain_drag,
        'gizmo_help': lambda: gizmo_help,
        'gpu': lambda: gpu,
        'help_layout': lambda: help_layout,
        'hit_handle': lambda: hit_handle,
        'hover_cursor': lambda: hover_cursor,
        'in_rect': lambda: in_rect,
        'keyframe_drag': lambda: keyframe_drag,
        'layout': lambda: layout,
        'lines': lambda: lines,
        'locus': lambda: locus,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'marker': lambda: marker,
        'math': lambda: math,
        'native': lambda: native,
        'nodes': lambda: nodes,
        'pointer_event': lambda: pointer_event,
        'pointer_region': lambda: pointer_region,
        're': lambda: re,
        'rectangle': lambda: rectangle,
        'reference_hint': lambda: reference_hint,
        'register': lambda: register,
        'release_drag_references': lambda: release_drag_references,
        'remove_node_group': lambda: remove_node_group,
        'remove_timer': lambda: remove_timer,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'scope_consumers': lambda: scope_consumers,
        'scope_gpu_role': lambda: scope_gpu_role,
        'scope_retry_hit': lambda: scope_retry_hit,
        'scope_status_target': lambda: scope_status_target,
        'set_hover_cursor': lambda: set_hover_cursor,
        'settle_drags': lambda: settle_drags,
        'shader': lambda: shader,
        'text': lambda: text,
        'theme': lambda: theme,
        'time': lambda: time,
        'tone_layout': lambda: tone_layout,
        'tr': lambda: tr,
        'triangles': lambda: triangles,
        'unicodedata': lambda: unicodedata,
        'unregister': lambda: unregister,
        'warp_view': lambda: warp_view,
        'wrap_help': lambda: wrap_help,
    }, writers={
        'COLORS': _set_export_COLORS,
        'LOG': _set_export_LOG,
        'MUTED': _set_export_MUTED,
        'TEXT': _set_export_TEXT,
        '_active_drags': _set_export__active_drags,
        '_classes': _set_export__classes,
        '_drag_timers': _set_export__drag_timers,
        '_handle': _set_export__handle,
        '_hover_cursors': _set_export__hover_cursors,
        '_last_error': _set_export__last_error,
        '_registered': _set_export__registered,
        '_routers': _set_export__routers,
        '_shaders': _set_export__shaders,
    'shader': _set_export_shader, 'lines': _set_export_lines, 'triangles': _set_export_triangles, 'CachedTriangles': _set_export_CachedTriangles, 'ColoredMesh': _set_export_ColoredMesh, 'colored_triangles': _set_export_colored_triangles, 'flush_color': _set_export_flush_color, 'end_color_drag': _set_export_end_color_drag, 'keyframe_drag': _set_export_keyframe_drag, 'release_drag_references': _set_export_release_drag_references, 'finish_drag': _set_export_finish_drag, 'settle_drags': _set_export_settle_drags, 'cleanup_drag_timers': _set_export_cleanup_drag_timers, 'set_hover_cursor': _set_export_set_hover_cursor, 'cleanup_hover_cursors': _set_export_cleanup_hover_cursors, 'scope_status_target': _set_export_scope_status_target, 'scope_retry_hit': _set_export_scope_retry_hit, 'scope_gpu_role': _set_export_scope_gpu_role, 'comparison_hit': _set_export_comparison_hit, '_redraw_rail_hover': _set_export__redraw_rail_hover, 'clear_rail_hover': _set_export_clear_rail_hover, 'hover_cursor': _set_export_hover_cursor, '_hover_cursor': _set_export__hover_cursor, 'rectangle': _set_export_rectangle, 'circle': _set_export_circle, 'text': _set_export_text, 'wrap_help': _set_export_wrap_help, 'help_layout': _set_export_help_layout, 'draw_hover_help': _set_export_draw_hover_help, 'arrow': _set_export_arrow, 'scope_consumers': _set_export_scope_consumers, 'layout': _set_export_layout, 'dashed_line': _set_export_dashed_line, 'draw_value_slider': _set_export_draw_value_slider, 'draw_exposure': _set_export_draw_exposure, 'draw_gain_controls': _set_export_draw_gain_controls, 'curve': _set_export_curve, 'reference_hint': _set_export_reference_hint, 'draw_footer': _set_export_draw_footer, 'draw_board': _set_export_draw_board, 'draw_header': _set_export_draw_header, 'draw': _set_export_draw, 'PG_OT_diagram_drag': _set_export_PG_OT_diagram_drag, 'pointer_region': _set_export_pointer_region, 'pointer_event': _set_export_pointer_event, 'PG_OT_diagram_events': _set_export_PG_OT_diagram_events, 'ensure_routers': _set_export_ensure_routers, 'register': _set_export_register, 'unregister': _set_export_unregister})
