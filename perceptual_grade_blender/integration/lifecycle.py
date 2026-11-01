"""Exact product Session recovery and ordered native teardown.

RNA registrations retain their original Session across module reload. Cleanup
uses explicit component references and never discovers replacement module state.
"""
import logging
import bpy
from ...blender_runtime.execution.diagnostics import log_exception
from .components import Component, component_get

LOG = logging.getLogger(__package__)
OWNER = 'perceptual_grade'
PACKAGE = __package__.rsplit('.', 1)[0]
RELEASE = '1.21.0'
STATE_ATTRIBUTE = '_perceptual_grade_registration'
MODULES = frozenset(('graph.bake', 'graph.nodes', 'graph.grades', 'graph.stacks', 'graph.editing', 'graph.output', 'ui.panel', 'ui.diagram', 'ui.help', 'scopes.automatic', 'scopes.sampling', 'scopes.images', 'scopes.capture', 'scopes.context', 'scopes.bake_routes', 'scopes.color', 'scopes.jobs', 'graph.lut', 'graph.gpu', 'ui.locus', 'scopes.reference', 'graph.compositor', 'graph.scheduler', 'ui.warp_edit'))
SPECS = (
    ('Operator', 'PERCEPTUAL_GRADE_OT_save_guarded'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_save_as_guarded'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_render_guarded'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_render_animation_guarded'),
    ('PropertyGroup', 'PGLinearSettings'),
    ('PropertyGroup', 'PGToneSettings'),
    ('PropertyGroup', 'PGChromaticSettings'),
    ('PropertyGroup', 'PGStackFocus'),
    ('PropertyGroup', 'PGStackEntry'),
    ('NodeTree', 'PGGradeData'),
    ('Node', 'PGShaderGrade'),
    ('Node', 'PGCompositorGrade'),
    ('Node', 'PGShaderStack'),
    ('Node', 'PGCompositorStack'),
    ('Node', 'PGCompositorBake'),
    ('Node', 'PGShaderBake'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_stack_add'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_stack_remove'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_stack_up'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_stack_down'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_stack_edit'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_add_stack'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_new_grade'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_unlink_grade'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_prepare_output'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_bake_compositor'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_add_bake'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_add_linear'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_linear'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_white'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_directions'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_toggle_diagram'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_set_peak'),
    ('Menu', 'PG_MT_peak'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_tone'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_chromatic'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_warp_select'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_reset_stage'),
    ('Panel', 'PG_PT_main'),
    ('Panel', 'PG_PT_linear'),
    ('Panel', 'PG_PT_remap'),
    ('Panel', 'PG_PT_chromatic'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_diagram_drag'),
    ('Operator', 'PERCEPTUAL_GRADE_OT_diagram_events'),
)
HANDLERS = ('depsgraph_update_pre', 'depsgraph_update_post', 'frame_change_pre', 'frame_change_post', 'animation_playback_pre', 'animation_playback_post', 'render_init', 'render_pre', 'render_post', 'render_complete', 'render_cancel', 'save_pre',
            'load_pre', 'undo_pre', 'redo_pre', 'load_post', 'undo_post', 'redo_post', 'save_post', 'save_post_fail', 'composite_pre', 'composite_post', 'composite_cancel')
WM_PROPERTIES = ('pg_diagram_enabled', 'pg_diagram_size', 'pg_diagram_x',
                 'pg_diagram_y', 'pg_stage','pg_tone_width','pg_selected_tone','pg_color_width')
def owned_module(name):
    if not isinstance(name, str):
        return False
    prefix = PACKAGE + '.'
    return name.startswith(prefix) and name[len(prefix):] in MODULES


def owned(value):
    return owned_module(getattr(value, '__module__', ''))


def live_classes(strict=False):
    result = []
    for base, identifier in SPECS:
        cls = getattr(bpy.types, base).bl_rna_get_subclass_py(identifier, None)
        if cls is not None:
            if owned(cls):
                result.append(cls)
            elif strict:
                from ..localization.i18n import create as create_i18n
                raise RuntimeError(create_i18n(None).tr('error.registration_conflict', identifier=identifier))
    return result


def unique(values):
    result, seen = [], set()
    for value in values:
        if value is not None and id(value) not in seen:
            seen.add(id(value));result.append(value)
    return result


def menu_callbacks():
    # Blender's append/remove API stores appended callbacks on this function.
    return [fn for menu_type in (bpy.types.NODE_MT_category_shader_color,bpy.types.NODE_MT_category_compositor_color)
            for fn in getattr(menu_type.draw, '_draw_funcs', ())
            if owned(fn) and fn.__name__ == 'menu']



def registered_session():
    classes = live_classes(strict=True)
    sessions = unique(getattr(cls, STATE_ATTRIBUTE, None) for cls in classes)
    if classes and (len(sessions) != 1 or any(getattr(cls, STATE_ATTRIBUTE, None) is not sessions[0] for cls in classes)):
        raise RuntimeError('Native registration lost its exact original Session')
    return sessions[0] if sessions else None


def cleanup(strict=False):
    session = registered_session()
    return True if session is None else session.close()


def create(_application):
    ownership = _application.component('ownership')
    tr = _application.component('localization.i18n').tr
    remove_keymap_item = _application.app.blender.native.remove_keymap_item
    _snapshot = None
    _cleanup_pending = False
    _cleanup_failures = 0
    _cleanup_timers = set()

    def finish_native_cleanup():
        """A scripted disable must wait until native rendering releases RNA."""
        output = _application.component('graph.output'); output_guard = _application.component('graph.output_guard')
        output_guard.reconcile(cleanup=True)
        if (output.native_running() or not output_guard.available(cleanup=True)
                or any(arena._borrows for arena in ownership.snapshots())):
            return .05
        cleanup()
        return None

    finish_native_cleanup = _application.component('integration.outcomes').callback(
        finish_native_cleanup, control='cleanup', fallback=.05,
        recovery='perceptual_grade.retry_recovery')


    def remember(nodes, ui, diagram):
        """Keep actual handles/functions/classes, not names that reload can replace."""
        nonlocal _snapshot
        scheduler = _application.component('graph.scheduler')
        output = _application.component('graph.output')
        gpu = _application.component('graph.gpu')
        _snapshot = dict(owner=OWNER, components=tuple(_application.components.values()),
                         timers=(nodes.poll_working_space, gpu._status_tick), draw_handles=(diagram._handle,),
                         keymaps=tuple(ui._keymaps), menus=(ui.menu,),
                         timer_registries=(scheduler._timers, _cleanup_timers),
                         output_unregister=output.unregister, arenas=ownership.snapshots())
        for cls in live_classes():
            setattr(cls, STATE_ATTRIBUTE, _application)


    def _attempt(action, *args, **kwargs):
        """Best-effort teardown is exhaustive, with each failure recorded."""
        nonlocal _cleanup_failures
        try:
            action(*args, **kwargs)
            return True
        except Exception:
            _cleanup_failures += 1
            log_exception(LOG, 'Could not clean up Perceptual Grade resource: %s',
                          getattr(action, '__name__', action))
            return False


    def _remove_keymap_items(keymaps):
        """Consume duplicate registrations before removing each native item once."""
        pending, seen = [], set()
        for keymap, item in keymaps:
            try:
                identity = (int(keymap.as_pointer()), int(item.as_pointer()))
            except (ReferenceError, RuntimeError):
                continue
            if identity not in seen:
                seen.add(identity)
                pending.append((keymap,item))
        keymaps.clear()
        for keymap, item in pending:
            try:
                if any(candidate == item for candidate in keymap.keymap_items):
                    remove_keymap_item(keymap,item)
            except (ReferenceError, RuntimeError):
                pass  # Keymap may have been replaced when loading another file.


    def cleanup(strict=False):
        """Remove only Perceptual Grade resources; keep native graphs and node ID values."""
        nonlocal _snapshot, _cleanup_pending, _cleanup_failures
        failures_before = _cleanup_failures
        classes = [cls for cls in _application.classes if getattr(cls, 'is_registered', False)]
        grade_module = _application.components.get('graph.grades')
        snapshots = [_snapshot] if _snapshot is not None else []
        _application.closing = True
        _runtime_owner = _application.app.execution.current_runtime()
        snapshots = [s for s in snapshots if isinstance(s, dict) and s.get('owner') == OWNER]
        # A live synchronous borrow is a native reader even when no guarded
        # output operation owns it. Retain exact registrations until it exits.
        if any(arena._borrows for arena in ownership.snapshots()):
            _application.waiting_for_readers = True
            _cleanup_pending = True
            _application.app.execution.fence()
            if classes and not bpy.app.timers.is_registered(finish_native_cleanup):
                bpy.app.timers.register(finish_native_cleanup, first_interval=.05)
                _cleanup_timers.add(finish_native_cleanup)
            return False
        # Packing and portable conversion touch live output resources. Keep the
        # registration intact until native readers have finished, including jobs
        # that were not launched through our guarded render operation.
        if classes and hasattr(bpy.data, 'window_managers'):
            output_guard = _application.component('graph.output_guard')
            output_guard.reconcile(cleanup=True)
            if not output_guard.available(cleanup=True):
                _application.waiting_for_readers = True
                _cleanup_pending = True
                for runtime in unique([_runtime_owner] + [s.get('runtime') for s in snapshots]):
                    runtime.fenced = True
                    runtime.cleanup_blocked = True
                if not bpy.app.timers.is_registered(finish_native_cleanup):
                    bpy.app.timers.register(finish_native_cleanup, first_interval=.05)
                    _cleanup_timers.add(finish_native_cleanup)
                return False
        _application.waiting_for_readers = False
        if _application.app.fenced:
            # Native recovery already reached App.close in a preceding pass.
            # Retry its exact pending ledger without reopening Store admission.
            if not _attempt(_application.app.close):
                return False
            for cls in reversed(classes):
                if not _attempt(_application.unregister_class, cls):
                    return False
            _application.closed = True
            return True
        output_module = _application.components.get('graph.output')
        unregister_outputs = unique([s.get('output_unregister') for s in snapshots] +
                                    [getattr(output_module, 'unregister', None)])
        for unregister in unregister_outputs:
            try:
                if unregister() is False:
                    _cleanup_pending = True
                    for runtime in unique([_runtime_owner] + [s.get('runtime') for s in snapshots]):
                        runtime.fenced = True
                        runtime.cleanup_blocked = True
                    if not bpy.app.timers.is_registered(finish_native_cleanup):
                        bpy.app.timers.register(finish_native_cleanup, first_interval=.05)
                        _cleanup_timers.add(finish_native_cleanup)
                    return False
            except Exception:
                _cleanup_failures += 1
                log_exception(LOG, 'Could not unregister guarded output operations')
        retired_arenas = unique([arena for snapshot in snapshots for arena in snapshot.get("arenas", ())]
                               + list(ownership.snapshots()))
        try:
            _cleanup_pending = False
            for timer in tuple(_cleanup_timers):
                if bpy.app.timers.is_registered(timer):
                    _attempt(bpy.app.timers.unregister, timer)
            _cleanup_timers.clear()
            # output.unregister may have removed classes in the original snapshot.
            classes = [cls for cls in _application.classes if getattr(cls, 'is_registered', False)]
            runtimes = unique([_runtime_owner] + [s.get('runtime') for s in snapshots])
            components = list(_application.components.values())
            timers = [f for s in snapshots for f in s['timers']]
            # Keep the registry object itself: importlib.reload may replace the module's
            # current set, but Blender still owns the preceding callback objects.
            timer_registries=unique([registry for s in snapshots for registry in s.get('timer_registries',())])
            timers += [fn for registry in timer_registries for fn in registry]
            handles = [h for s in snapshots for h in s['draw_handles']]
            keymaps = [pair for s in snapshots for pair in s['keymaps']]
            for snapshot in snapshots:
                snapshot['keymaps'] = ()
            menus = [fn for s in snapshots for fn in s['menus']]
            panel = _application.components.get('ui.panel')
            if panel is not None: menus.append(panel.menu)
            bake = _application.components.get('graph.bake')
            if bake is not None: menus.append(bake.menu)
            # Retain exact callback identities across a module reload.
            callbacks = tuple(_application.handlers)
            # Exact lexical component instances belong to this Session, including
            # factories first used after the initial registration snapshot.
            # Invalidate sessions before any old callback can touch detached RNA. A
            # failed cleanup must not prevent subsequent independent resource cleanup.
            for g in components:
                if component_get(g, '__name__', '').endswith('.scheduler'):
                    if callable(component_get(g, 'lifecycle_event')):
                        _attempt(g['lifecycle_event'], 'disable')
                    if callable(component_get(g, 'recover')):
                        _attempt(g['recover'])
            for g in components:
                if callable(component_get(g, 'settle_drags')):
                    _attempt(g['settle_drags'])
            if grade_module is not None and classes:
                _attempt(grade_module.prepare_disable)
            for g in components:
                if not owned_module(component_get(g, '__name__', '')):
                    continue
                if callable(component_get(g, 'cleanup_drag_timers')):_attempt(g['cleanup_drag_timers'])
                if callable(component_get(g, 'cleanup_hover_cursors')):_attempt(g['cleanup_hover_cursors'])
                if component_get(g, '__name__','').endswith('.locus') and callable(component_get(g, 'clear')):_attempt(g['clear'])
                if component_get(g, '__name__','').endswith('.compositor') and callable(component_get(g, 'clear')):_attempt(g['clear'])
                if component_get(g, '__name__','').endswith('.graph.bake') and callable(component_get(g, 'clear')):_attempt(g['clear'])
                if component_get(g, '__name__','').endswith('.scheduler') and callable(component_get(g, 'shutdown')):_attempt(g['shutdown'])
                if component_get(g, '__name__','').endswith(('.scopes.automatic','.scopes.jobs','.scopes.capture','.scopes.context')) and callable(component_get(g, 'clear')):_attempt(g['clear'])
                clear_cubes=getattr(component_get(g, 'cube_nodes'),'clear',None)
                if component_get(g, '__name__','').endswith('.nodes') and callable(clear_cubes):_attempt(clear_cubes,release_gpu=True)
                if '_registered' in g: g['_registered'] = False
                if '_rendering' in g: g['_rendering'] = False
                timers.append(component_get(g, 'poll_working_space'))
                if component_get(g, '__name__','').endswith('.graph.gpu'):timers.append(component_get(g, '_status_tick'))
                handles.append(component_get(g, '_handle'))
                keymaps.extend(component_get(g, '_keymaps', ()))
                if '_keymaps' in g:
                    g['_keymaps'].clear()

            for fn in unique(timers):
                if bpy.app.timers.is_registered(fn):
                    _attempt(bpy.app.timers.unregister, fn)
            for registry in timer_registries:registry.clear()
            for name, fn in callbacks:
                collection = getattr(bpy.app.handlers, name)
                while fn in collection:
                    if not _attempt(collection.remove, fn):break
            for fn in unique(menus):
                for menu_type in (bpy.types.NODE_MT_category_shader_color,bpy.types.NODE_MT_category_compositor_color):
                    if fn in getattr(menu_type.draw, '_draw_funcs', ()):
                        _attempt(menu_type.remove, fn)
            for handle in unique(handles):
                try:
                    bpy.types.SpaceNodeEditor.draw_handler_remove(handle, 'WINDOW')
                except (ValueError, ReferenceError, RuntimeError):
                    pass  # Already removed by a partial-registration rollback.

            # Include installed keymap items if a reloaded module lost its Python list.
            config = bpy.context.window_manager.keyconfigs.addon
            if config and (classes or snapshots or callbacks):
                for km in config.keymaps:
                    keymaps.extend((km, item) for item in km.keymap_items
                                   if item.idname == 'perceptual_grade.diagram_drag')
            _remove_keymap_items(keymaps)

            if classes or snapshots:
                for name in ('pg_stack_entries','pg_working_white'):
                    if hasattr(bpy.types.Node,name):_attempt(delattr,bpy.types.Node,name)
                if hasattr(bpy.types.Node, 'pg_grade'):
                    _attempt(delattr,bpy.types.Node,'pg_grade')
                for name in ('pg_linear', 'pg_tone', 'pg_chromatic', 'pg_working_white'):
                    if hasattr(bpy.types.NodeTree, name):
                        _attempt(delattr,bpy.types.NodeTree,name)
                if hasattr(bpy.types.Node, 'pg_linear'):
                    prop = bpy.types.Node.bl_rna.properties.get('pg_linear')
                    if prop and prop.fixed_type.identifier == 'PGLinearSettings':
                        _attempt(delattr,bpy.types.Node,'pg_linear')
                if hasattr(bpy.types.Node,'pg_chromatic'):
                    prop=bpy.types.Node.bl_rna.properties.get('pg_chromatic')
                    if prop and prop.fixed_type.identifier=='PGChromaticSettings':_attempt(delattr,bpy.types.Node,'pg_chromatic')
                if hasattr(bpy.types.Node,'pg_tone'):
                    prop=bpy.types.Node.bl_rna.properties.get('pg_tone')
                    if prop and prop.fixed_type.identifier=='PGToneSettings':_attempt(delattr,bpy.types.Node,'pg_tone')
                for name in WM_PROPERTIES:
                    if hasattr(bpy.types.WindowManager, name):
                        _attempt(delattr,bpy.types.WindowManager,name)
            # Retire actual arenas retained by old registration snapshots as well as
            # the current module. This is independent of failed timer/cache cleanup.
            for arena in retired_arenas:
                _attempt(arena.detach)
            anchor = next((cls for cls in classes if cls.__name__ == 'PGLinearSettings'), None)
            for cls in reversed(classes):
                if cls is not anchor:
                    _attempt(_application.unregister_class, cls)

            for g in components:
                if not owned_module(component_get(g, '__name__', '')):
                    continue
                for name in ('_registered', '_class_registered', '_menu_registered', '_diagram_registered'):
                    if name in g:g[name] = False
                if '_handle' in g:g['_handle'] = None
                for name in ('_classes', '_keymaps', '_shaders', '_signatures', '_updating', '_routers'):
                    if name in g:g[name].clear()
                if component_get(g, '__name__','').endswith(('.scopes.sampling','.scopes.images','.scopes.color','.scopes.reference')) and callable(component_get(g, 'clear')):_attempt(g['clear'])
                if component_get(g, '__name__','').endswith('.graph.editing') and callable(component_get(g, 'clear')):_attempt(g['clear'])
            if grade_module is not None:
                _attempt(grade_module.clear)
            # All foreground views/graph cache references have now been fenced and
            # cleared. Shutdown does not access bpy.data and confirms child exits.
            output_lut = _application.component('graph.lut')
            _attempt(output_lut.clear, release_gpu=True)
            i18n = _application.component('localization.i18n')
            _attempt(i18n.unregister)
            transactions = _application.component('ownership.transactions')
            for runtime in runtimes:
                runtime.cleanup_blocked = (_cleanup_failures != failures_before or transactions.pending(cleanup=True))
                _attempt(runtime.shutdown)
            # Capture inputs outlive cancellation. Shutdown can retire the last
            # helper after its polling timer has been removed, so consume those
            # exact retirement receipts before completing foreground teardown.
            capture = _application.components.get('scopes.capture')
            if capture is not None:
                _attempt(capture.collect_retired)
            if _cleanup_failures != failures_before or transactions.pending(cleanup=True):
                return False
            # Complete these exact snapshot obligations while their actors are
            # alive. App.close retires actor storage; its successful return must
            # not be followed by another synchronous arena transition.
            for arena in retired_arenas:
                _attempt(arena.detach)
                _attempt(arena.fail)
            if _cleanup_failures != failures_before:
                return False
            retired_arenas = ()
            # This detached class carries only the exact Session while helper
            # shutdown is pending; it is the final native lifecycle anchor.
            if anchor is not None and not _attempt(_application.unregister_class, anchor):
                return False
            if not _attempt(_application.app.close):
                # Retain the same exact native anchor if terminal cleanup failed.
                if anchor is not None and not getattr(anchor, 'is_registered', False):
                    bpy.utils.register_class(anchor)
                return False
            _application.closed = True
            _snapshot = None
            return True
        finally:
            # Old registration snapshots can outlive teardown, including a failed
            # debt scan or registry cleanup. Drop raw cells and permanently fence
            # their controllers after the shutdown ledger has been inspected.
            for arena in retired_arenas:
                _attempt(arena.detach)
                _attempt(arena.fail)



    def _set_export_finish_native_cleanup(value):
        nonlocal finish_native_cleanup
        finish_native_cleanup = value

    def _set_export_remember(value):
        nonlocal remember
        remember = value

    def _set_export__attempt(value):
        nonlocal _attempt
        _attempt = value

    def _set_export_remove_keymap_item(value):
        nonlocal remove_keymap_item
        remove_keymap_item = value

    def _set_export__remove_keymap_items(value):
        nonlocal _remove_keymap_items
        _remove_keymap_items = value

    def _set_export_cleanup(value):
        nonlocal cleanup
        cleanup = value

    return Component({'cleanup': cleanup, 'remember': remember,
                      'finish_native_cleanup': finish_native_cleanup,
                      'remove_keymap_item': remove_keymap_item,
                      '_remove_keymap_items': _remove_keymap_items,
                      '__name__': __name__}, readers={
                          '_cleanup_pending': lambda: _cleanup_pending,
                          '_cleanup_failures': lambda: _cleanup_failures,
                          '_cleanup_timers': lambda: _cleanup_timers}, writers={'remove_keymap_item': _set_export_remove_keymap_item, 'finish_native_cleanup': _set_export_finish_native_cleanup, 'remember': _set_export_remember, '_attempt': _set_export__attempt, '_remove_keymap_items': _set_export__remove_keymap_items, 'cleanup': _set_export_cleanup})
