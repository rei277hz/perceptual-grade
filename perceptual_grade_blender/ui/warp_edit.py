"""Native Warp fields use the same validated transactions as overlay drags.

The window event router arms a snapshot before Blender's native number button
starts. The first proxy setter claims that snapshot. Blender still owns text
entry and undo; this adapter owns the coupled raw values and raw auto-key paths.
Release/confirmation is finalized after native button callbacks have run.
"""

from ..integration.components import Component

def create(_application):
    remove_node_group = _application.app.blender.native.remove_node_group
    from dataclasses import dataclass, field as member
    import math

    import bpy
    import numpy as np

    nodes = _application.component('graph.nodes'); grades = _application.component('graph.grades'); scheduler = _application.component('graph.scheduler')
    from ...perceptual_grade_core.model import color
    warp_scale = _application.component('ui.warp_scale'); warp_view = _application.component('ui.warp_view')
    _candidate = _application.component('ui.warp_drag')._candidate
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve; OwnedField = _application.component('ownership').OwnedField


    @dataclass(eq=False)
    class Edit:
        node_ref: object
        start: object
        tone: object
        space: object
        key: object
        context_ref: object = None
        field: str = ''
        model: object = None
        gesture: object = None
        view: object = None
        original_grade_ref: object = None
        touched: set = member(default_factory=set)
        cancelled: bool = False
        keyed: bool = False
        session: object = None

        @property
        def node(self):return resolve(self.node_ref)

        @property
        def context(self):return resolve(self.context_ref)

        @property
        def original_grade(self):return resolve(self.original_grade_ref)


    class EditorContext:
        """A saved editor context contains tokens, not borrowed native wrappers."""
        window = OwnedField('warp.edit.window')
        area = OwnedField('warp.edit.area')
        window_manager = OwnedField('warp.edit.window_manager')
        scene = OwnedField('warp.edit.scene')

        def __init__(self, context, area=None):
            self.window=context.window;self.area=area if area is not None else getattr(context,'area',None)
            self.window_manager=context.window_manager;self.scene=context.scene


    _edits = {}
    _pending = {}
    _promotions = {}


    def begin(node, context=None, key=None):
        """Arm a cheap snapshot; a sidebar click need not change any setting."""
        if key is None:
            key = context.window.as_pointer() if context is not None else node.as_pointer()
        if key in _edits:
            finish(_edits[key])
        saved = None if context is None else EditorContext(context)
        edit = Edit(retain(node,'warp.edit.node'), nodes.read_chromatic(node), nodes.read_tone(node),
                    nodes.grading_space(node), key, retain(saved,'warp.edit.context'),
                    original_grade_ref=retain(grades.selected(node),'warp.edit.grade'), session=scheduler.session())
        _edits[key] = edit
        return edit


    def _matches(edit, node):
        try:
            if edit.session != scheduler.session():
                return False
            return grades.data_owner(edit.node) == grades.data_owner(node)
        except (ReferenceError, RuntimeError):
            return False


    def _claim(edit, field):
        edit.field = field
        edit.model = color.model(edit.tone, edit.start, edit.space)
        edit.gesture = scheduler.begin(edit.node)
        # Numeric edits select the corresponding pad handle without hiding any
        # sidebar rows. The same editor's view stays frozen until confirmation.
        ui = edit.node.pg_chromatic
        if field.startswith('web_'):
            _, _, tone, handle = field.split('_')
            ui.active_tone, ui.active_handle = int(tone), int(handle)
        elif field.startswith(('adaptance_a_', 'adaptance_b_')):
            ui.active_tone, ui.active_handle = int(field.rsplit('_', 1)[1]), -1
        context = edit.context
        if context is not None and getattr(context.window_manager, 'pg_diagram_enabled', False):
            warp = _application.component('ui.warp')
            region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
            if region is not None:
                editor = (context.window.as_pointer(), context.area.as_pointer(), region.as_pointer())
                edit.view = warp_view.begin(editor, edit.node, ui.active_tone,
                                             warp.view_radius(edit.model, ui.active_tone))


    def edit_sidebar(node, field, value):
        """Commit an accepted raw edit, preserving saved property/F-curve names.

        Direct Python assignment is a single transaction. Native repeated setter
        calls use the pre-gesture baseline, including the original Chroma Stops.
        """
        if field not in color.PARAM_NAMES or not field.startswith(('web_', 'adaptance_')):
            return {}
        if not nodes.editable(node) or not math.isfinite(value):
            return {}
        edit = next((e for e in _edits.values() if _matches(e, node)), None)
        if edit is not None and edit.cancelled:
            return {}
        current = nodes.read_chromatic(node)
        # Avoid allocating a Grade or creating a scheduler gesture for a no-op.
        if value == getattr(current, field) and (edit is None or not edit.field):
            return {}
        try:
            if edit is not None:
                if edit.field and edit.field != field:
                    finish(edit)
                    edit = None
                elif not edit.field:
                    _claim(edit, field)
            model = edit.model if edit is not None else nodes.color_model(node)
            if field.startswith('web_scale_'):
                _, _, tone, handle = field.split('_')
                values = warp_scale.edit(model, int(tone), int(handle), value)
            else:
                values = {field: float(value)}
                _candidate(model, values)
                _candidate(model, {field: float(np.float32(value))})
            if not values:
                return {}
            if edit is not None:
                edit.touched.update(values)
            changed = {name: value for name, value in values.items()
                       if float(np.float32(value)) != getattr(current, name)}
            if changed:
                nodes.write_chromatic(edit.node if edit is not None else node, changed)
            if edit is not None and _pending.get(edit):
                # Text confirmation can deliver its last setter after the router
                # saw Enter. Key the final raw pair before native UI pushes undo.
                _keyframe(edit)
            return values
        except (ValueError, ArithmeticError, ReferenceError, RuntimeError):
            return {}


    def _keyframe(edit):
        if edit.context is not None and edit.context.scene.tool_settings.use_keyframe_insert_auto:
            current = nodes.read_chromatic(edit.node)
            for name in sorted(edit.touched):
                if getattr(current, name) != getattr(edit.start, name):
                    nodes.keyframe_chromatic(edit.node, name)
        edit.keyed = True


    def finish(edit, commit=True, strict=False):
        """Settle after native callbacks; cancellation restores exact stored values."""
        if edit is None or _edits.get(edit.key) is not edit:
            return
        _edits.pop(edit.key, None)
        _pending.pop(edit, None)
        _promotions.pop(edit.key, None)
        if edit.session != scheduler.session():
            scheduler.end(edit.gesture)
            warp_view.end(edit.view)
            return
        commit = commit and not edit.cancelled
        try:
            if not edit.touched:
                return
            if not commit:
                nodes.write_chromatic(edit.node, {name: getattr(edit.start, name) for name in edit.touched})
                if edit.original_grade is None and grades.selected(edit.node) is not None:
                    grade = grades.selected(edit.node)
                    grades.unlink(edit.node)
                    if grade.users == int(grade.use_fake_user):
                        remove_node_group(grade)
            elif not edit.keyed:
                _keyframe(edit)
        except (ReferenceError, RuntimeError):
            if strict:
                raise
        finally:
            scheduler.end(edit.gesture)
            warp_view.end(edit.view)
            if edit.touched and commit:
                jobs = _application.component('scopes.jobs')
                try:jobs.finish_edit(edit.node)
                except (ReferenceError, RuntimeError):pass


    def flush():
        """One event-loop turn, never a pause-based gesture timeout."""
        for edit, commit in tuple(_pending.items()):
            finish(edit, commit)
        return None


    def defer_finish(edit, commit=True):
        if (edit is None or _edits.get(edit.key) is not edit or
                edit.session != scheduler.session()):
            return
        if not commit:
            edit.cancelled = True
        _pending[edit] = commit
        if commit and edit.touched:
            _keyframe(edit)
        scheduler.wake(flush, delay=0.)


    def promote_router():
        """Observe before Blender's native button modal handler consumes release.

        Number/text buttons insert their UI handler during the initiating press.
        Reinstalling our pass-through router on the following event-loop turn puts
        it ahead of that handler, without starting an operator in an RNA setter.
        """
        diagram = _application.component('ui.diagram')
        for key, edit in tuple(_promotions.items()):
            _promotions.pop(key, None)
            if _edits.get(key) is not edit or edit.context is None or not diagram._registered:
                continue
            try:
                window = edit.context.window
                diagram._routers.pop(key, None)
                with bpy.context.temp_override(window=window):
                    bpy.ops.perceptual_grade.diagram_events('INVOKE_DEFAULT')
            except (ReferenceError, RuntimeError):
                finish(edit, False)
        return None


    def active(window):
        return window.as_pointer() in _edits


    def sidebar_node(context, event):
        """Find an editable Warp sidebar even when the overlay is closed."""
        in_rect = _application.component('ui.geometry').in_rect
        for area in context.window.screen.areas:
            if area.type != 'NODE_EDITOR':
                continue
            for region in area.regions:
                if (region.type != 'UI' or region.width <= 1 or
                        not in_rect((event.mouse_x, event.mouse_y), (region.x, region.y, region.width, region.height))):
                    continue
                with context.temp_override(area=area, region=region):
                    node = nodes.active_grade(context)
                    if node is not None and nodes.editable(node) and node.pg_chromatic.family == 'WARP':
                        # bpy.context is dynamic; retain only stable RNA objects.
                        saved = EditorContext(context,area)
                        return node, saved
        return None


    def observe_event(context, event):
        """Passively pair native numeric callbacks with their pointer/key gesture."""
        key = context.window.as_pointer()
        edit = _edits.get(key)
        if event.type == 'LEFTMOUSE' and event.value in {'PRESS', 'DOUBLE_CLICK'}:
            # A native tool can consume its final event. A fresh press always
            # settles the previous edit before capturing a new frozen baseline.
            if edit is not None:
                finish(edit)
            found = sidebar_node(context, event)
            if found is not None:
                edit = begin(*found, key=key)
                if not bpy.app.background:
                    _promotions[key] = edit
                    scheduler.wake(promote_router, delay=0.)
        elif edit is not None:
            if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                defer_finish(edit, False)
            elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and edit.field:
                defer_finish(edit)
            elif event.type in {'RET', 'NUMPAD_ENTER', 'TAB'} and event.value == 'PRESS':
                defer_finish(edit)
            elif event.type == 'WINDOW_DEACTIVATE':
                defer_finish(edit)


    def settle(commit=False):
        """Commit on save, otherwise discard without writing old snapshot values."""
        from ...blender_runtime.execution.diagnostics import Failure
        scheduler.unwake(flush)
        scheduler.unwake(promote_router)
        errors = []
        for edit in tuple(_edits.values()):
            try:
                if commit:
                    finish(edit, True, strict=True)
                else:
                    scheduler.end(edit.gesture)
                    warp_view.end(edit.view)
            except Exception as exc:
                errors.append(Failure.capture(exc))
            finally:
                _edits.pop(edit.key, None)
                _pending.pop(edit, None)
        _pending.clear()
        _promotions.clear()
        if errors:
            raise errors[0].exception('Could not settle a Perceptual Grade sidebar edit')


    def clear():
        """Discard transient snapshots on undo/load/unregister; never replay them."""
        settle(commit=False)

    def _set_export__edits(value):
        nonlocal _edits
        _edits = value

    def _set_export__pending(value):
        nonlocal _pending
        _pending = value

    def _set_export__promotions(value):
        nonlocal _promotions
        _promotions = value

    def _set_export_Edit(value):
        nonlocal Edit
        Edit = value

    def _set_export_EditorContext(value):
        nonlocal EditorContext
        EditorContext = value

    def _set_export_begin(value):
        nonlocal begin
        begin = value

    def _set_export__matches(value):
        nonlocal _matches
        _matches = value

    def _set_export__claim(value):
        nonlocal _claim
        _claim = value

    def _set_export_edit_sidebar(value):
        nonlocal edit_sidebar
        edit_sidebar = value

    def _set_export__keyframe(value):
        nonlocal _keyframe
        _keyframe = value

    def _set_export_finish(value):
        nonlocal finish
        finish = value

    def _set_export_flush(value):
        nonlocal flush
        flush = value

    def _set_export_defer_finish(value):
        nonlocal defer_finish
        defer_finish = value

    def _set_export_promote_router(value):
        nonlocal promote_router
        promote_router = value

    def _set_export_active(value):
        nonlocal active
        active = value

    def _set_export_sidebar_node(value):
        nonlocal sidebar_node
        sidebar_node = value

    def _set_export_observe_event(value):
        nonlocal observe_event
        observe_event = value

    def _set_export_settle(value):
        nonlocal settle
        settle = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Edit': lambda: Edit,
        'EditorContext': lambda: EditorContext,
        'OwnedField': lambda: OwnedField,
        '_candidate': lambda: _candidate,
        '_claim': lambda: _claim,
        '_edits': lambda: _edits,
        '_keyframe': lambda: _keyframe,
        '_matches': lambda: _matches,
        '_pending': lambda: _pending,
        '_promotions': lambda: _promotions,
        'active': lambda: active,
        'begin': lambda: begin,
        'bpy': lambda: bpy,
        'clear': lambda: clear,
        'color': lambda: color,
        'dataclass': lambda: dataclass,
        'defer_finish': lambda: defer_finish,
        'edit_sidebar': lambda: edit_sidebar,
        'finish': lambda: finish,
        'flush': lambda: flush,
        'grades': lambda: grades,
        'math': lambda: math,
        'member': lambda: member,
        'nodes': lambda: nodes,
        'np': lambda: np,
        'observe_event': lambda: observe_event,
        'promote_router': lambda: promote_router,
        'remove_node_group': lambda: remove_node_group,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'scheduler': lambda: scheduler,
        'settle': lambda: settle,
        'sidebar_node': lambda: sidebar_node,
        'warp_scale': lambda: warp_scale,
        'warp_view': lambda: warp_view,
    }, writers={
        '_edits': _set_export__edits,
        '_pending': _set_export__pending,
        '_promotions': _set_export__promotions,
    'Edit': _set_export_Edit, 'EditorContext': _set_export_EditorContext, 'begin': _set_export_begin, '_matches': _set_export__matches, '_claim': _set_export__claim, 'edit_sidebar': _set_export_edit_sidebar, '_keyframe': _set_export__keyframe, 'finish': _set_export_finish, 'flush': _set_export_flush, 'defer_finish': _set_export_defer_finish, 'promote_router': _set_export_promote_router, 'active': _set_export_active, 'sidebar_node': _set_export_sidebar_node, 'observe_event': _set_export_observe_event, 'settle': _set_export_settle, 'clear': _set_export_clear})
