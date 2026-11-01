"""Compositor scheduling and cached, pre-composite Render Layers scope input."""

from ..integration.components import Component

def create(_application):
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text
    from dataclasses import dataclass
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    from pathlib import Path
    from collections import OrderedDict
    import logging
    import tempfile
    import bpy
    from bpy.app.handlers import persistent
    scheduler = _application.component('graph.scheduler')


    LOG=logging.getLogger(__package__)
    _snapshots=OrderedDict()
    _samples={}
    _pending=OrderedDict()
    _revisions={}
    _errors={}
    _temp=None
    _busy=False


    def preview_running():return scheduler.preview_running()

    def can_commit(node):return scheduler.can_commit(node)

    def preview_installed(node):scheduler.preview_installed(node)


    @persistent
    def composite_pre(scene,*_args):
        output_guard = _application.component('graph.output_guard')
        with output_guard.native_callback():
            output_guard.evaluation_started('composite',scene.session_uid)
            scheduler.frame_start(scene)


    @persistent
    def composite_post(scene,*_args):
        output_guard = _application.component('graph.output_guard')
        with output_guard.native_callback():
            try:scheduler.frame_end(scene)
            finally:output_guard.evaluation_finished('composite',scene.session_uid)


    @persistent
    def composite_cancel(scene,*_args):
        output_guard = _application.component('graph.output_guard')
        with output_guard.native_callback():
            try:scheduler.frame_end(scene,cancelled=True)
            finally:output_guard.evaluation_finished('composite',scene.session_uid)


    def defer_update(node,force=False):return scheduler.defer_update(node,force)

    def flush_update(node):scheduler.flush_update(node)

    def update_tick():return scheduler.update_tick()


    def tag_output(node):
        """Invalidate the evaluated group and every scene root that contains it."""
        tree=node.id_data
        tree.update_tag()
        if tree.bl_idname!='CompositorNodeTree':return
        lut = _application.component('graph.lut')
        scheduler._commits[node.as_pointer()]=(tree.as_pointer(),lut.signal_signature(node))
        node.node_tree.update_tag()
        for scene in bpy.data.scenes:
            root=scene.compositing_node_group
            if root and (root==tree or root.contains_tree(tree)):
                root.update_tag();scene.update_tag()
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type in {'NODE_EDITOR','IMAGE_EDITOR'}:area.tag_redraw()


    def scene_for_tree(tree):
        candidates=[scene for scene in bpy.data.scenes if scene.compositing_node_group and
                    (scene.compositing_node_group==tree or scene.compositing_node_group.contains_tree(tree))]
        if bpy.context.scene in candidates:return bpy.context.scene
        if len(candidates)==1:return candidates[0]
        raise ValueError('Choose the scene that uses this compositor graph')


    @dataclass(frozen=True,init=False)
    class RenderLayerSource:
        scene_ref: object
        layer: str
        pass_name: str='Combined'
        view: str=''

        def __init__(self,scene,layer,pass_name='Combined',view=''):
            object.__setattr__(self,'scene_ref',retain(scene,'compositor.source'))
            object.__setattr__(self,'layer',layer)
            object.__setattr__(self,'pass_name',pass_name)
            object.__setattr__(self,'view',view)

        @property
        def scene(self):return resolve(self.scene_ref)


    def _key(scene):return scene.as_pointer(),scene.frame_current,_revisions.get(scene.as_pointer(),0)


    def invalidate_render(scene):
        """Called once when a render completes, never on grading-only changes."""
        pointer=scene.as_pointer();_revisions[pointer]=_revisions.get(pointer,0)+1
        for key in tuple(_snapshots):
            if key[0]==pointer:
                _snapshots.pop(key).unlink(missing_ok=True)
        for key in tuple(_samples):
            if key[0][0]==pointer:_samples.pop(key,None)
        _errors.pop(pointer,None)


    def capture(scene):
        """Snapshot raw passes, retaining their scene-linear working RGB.

        Runs outside draw callbacks. Saving a multilayer Render Result keeps
        Combined passes separate from the downstream Composite result. The small
        sampled grids are reused throughout every subsequent grading drag.
        """
        nonlocal _temp, _busy
        image=bpy.data.images.get('Render Result')
        if image is None or not image.has_data:
            raise ValueError('Render the scene to populate Render Layers scopes')
        if _temp is None:_temp=tempfile.TemporaryDirectory(prefix='perceptual-grade-scopes-')
        key=_key(scene);path=Path(_temp.name)/f'{key[0]}-{key[1]}-{key[2]}.exr'
        settings=scene.render.image_settings
        fields=('media_type','file_format','color_mode','color_depth','exr_codec','use_exr_interleave','color_management')
        old={name:getattr(settings,name) for name in fields}
        old_space=settings.linear_colorspace_settings.name
        _busy=True
        try:
            settings.media_type='MULTI_LAYER_IMAGE';settings.file_format='OPEN_EXR_MULTILAYER'
            settings.color_mode='RGBA';settings.color_depth='32';settings.exr_codec='ZIPS'
            settings.use_exr_interleave=True  # One scanline part for our small-grid reader.
            settings.color_management='OVERRIDE';settings.linear_colorspace_settings.name='scene_linear'
            image.save_render(str(path),scene=scene)
        except Exception:
            path.unlink(missing_ok=True);raise
        finally:
            for name,value in old.items():setattr(settings,name,value)
            settings.linear_colorspace_settings.name=old_space
            _busy=False
        _snapshots[key]=path
        while len(_snapshots)>2:
            expired,file=_snapshots.popitem(last=False);file.unlink(missing_ok=True)
            for sample_key in tuple(_samples):
                if sample_key[0]==expired:_samples.pop(sample_key,None)
        return key


    def tick():
        if not _pending:return None
        if bpy.app.is_job_running('RENDER') or bpy.app.is_job_running('COMPOSITE'):return .15
        pointer,scene_ref=_pending.popitem(last=False)
        scene=resolve(scene_ref)
        if scene is None:return .1 if _pending else None
        try:
            capture(scene);_errors.pop(pointer,None)
        except (ValueError,RuntimeError,ReferenceError,OSError) as exc:
            _errors[pointer]=diagnostic_text(exc);LOG.warning('Perceptual Grade compositor scopes: {error}'.format(error=exc))
        cube_nodes = _application.component('graph.lut')
        cube_nodes.redraw()
        return .1 if _pending else None


    def sample(source,size=None):
        scope_exr = _application.component('scopes.exr')
        scene=source.scene
        if scene is None:raise ReferenceError('Compositor scope scene has been retired')
        key=_key(scene);pointer=key[0]
        if key not in _snapshots:
            if pointer in _errors:raise ValueError(_errors[pointer])
            _pending[pointer]=retain(scene,'compositor.pending_scene')
            scheduler.wake(tick,.01)
            raise ValueError('Preparing compositor scopes')
        sample_key=(key,source.layer,source.pass_name,source.view,size)
        if sample_key not in _samples:
            _samples[sample_key]=scope_exr.sample(_snapshots[key],source.layer,source.pass_name,size,source.view)
        return _samples[sample_key],sample_key


    def clear():
        nonlocal _temp, _busy
        scheduler.unwake(tick)
        scheduler.unwake(update_tick)
        scheduler.clear_compositor()
        _snapshots.clear();_samples.clear();_pending.clear();_revisions.clear();_errors.clear()
        if _temp is not None:_temp.cleanup();_temp=None
        _busy=False

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export__busy(value):
        nonlocal _busy
        _busy = value

    def _set_export__errors(value):
        nonlocal _errors
        _errors = value

    def _set_export__pending(value):
        nonlocal _pending
        _pending = value

    def _set_export__revisions(value):
        nonlocal _revisions
        _revisions = value

    def _set_export__samples(value):
        nonlocal _samples
        _samples = value

    def _set_export__snapshots(value):
        nonlocal _snapshots
        _snapshots = value

    def _set_export__temp(value):
        nonlocal _temp
        _temp = value

    def _set_export_preview_running(value):
        nonlocal preview_running
        preview_running = value

    def _set_export_can_commit(value):
        nonlocal can_commit
        can_commit = value

    def _set_export_preview_installed(value):
        nonlocal preview_installed
        preview_installed = value

    def _set_export_composite_pre(value):
        nonlocal composite_pre
        composite_pre = value

    def _set_export_composite_post(value):
        nonlocal composite_post
        composite_post = value

    def _set_export_composite_cancel(value):
        nonlocal composite_cancel
        composite_cancel = value

    def _set_export_defer_update(value):
        nonlocal defer_update
        defer_update = value

    def _set_export_flush_update(value):
        nonlocal flush_update
        flush_update = value

    def _set_export_update_tick(value):
        nonlocal update_tick
        update_tick = value

    def _set_export_tag_output(value):
        nonlocal tag_output
        tag_output = value

    def _set_export_scene_for_tree(value):
        nonlocal scene_for_tree
        scene_for_tree = value

    def _set_export_RenderLayerSource(value):
        nonlocal RenderLayerSource
        RenderLayerSource = value

    def _set_export__key(value):
        nonlocal _key
        _key = value

    def _set_export_invalidate_render(value):
        nonlocal invalidate_render
        invalidate_render = value

    def _set_export_capture(value):
        nonlocal capture
        capture = value

    def _set_export_tick(value):
        nonlocal tick
        tick = value

    def _set_export_sample(value):
        nonlocal sample
        sample = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'LOG': lambda: LOG,
        'OrderedDict': lambda: OrderedDict,
        'Path': lambda: Path,
        'RenderLayerSource': lambda: RenderLayerSource,
        '_busy': lambda: _busy,
        '_errors': lambda: _errors,
        '_key': lambda: _key,
        '_pending': lambda: _pending,
        '_revisions': lambda: _revisions,
        '_samples': lambda: _samples,
        '_snapshots': lambda: _snapshots,
        '_temp': lambda: _temp,
        'bpy': lambda: bpy,
        'can_commit': lambda: can_commit,
        'capture': lambda: capture,
        'clear': lambda: clear,
        'composite_cancel': lambda: composite_cancel,
        'composite_post': lambda: composite_post,
        'composite_pre': lambda: composite_pre,
        'dataclass': lambda: dataclass,
        'defer_update': lambda: defer_update,
        'diagnostic_text': lambda: diagnostic_text,
        'flush_update': lambda: flush_update,
        'invalidate_render': lambda: invalidate_render,
        'logging': lambda: logging,
        'persistent': lambda: persistent,
        'preview_installed': lambda: preview_installed,
        'preview_running': lambda: preview_running,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'sample': lambda: sample,
        'scene_for_tree': lambda: scene_for_tree,
        'scheduler': lambda: scheduler,
        'tag_output': lambda: tag_output,
        'tempfile': lambda: tempfile,
        'tick': lambda: tick,
        'update_tick': lambda: update_tick,
    }, writers={
        'LOG': _set_export_LOG,
        '_busy': _set_export__busy,
        '_errors': _set_export__errors,
        '_pending': _set_export__pending,
        '_revisions': _set_export__revisions,
        '_samples': _set_export__samples,
        '_snapshots': _set_export__snapshots,
        '_temp': _set_export__temp,
    'preview_running': _set_export_preview_running, 'can_commit': _set_export_can_commit, 'preview_installed': _set_export_preview_installed, 'composite_pre': _set_export_composite_pre, 'composite_post': _set_export_composite_post, 'composite_cancel': _set_export_composite_cancel, 'defer_update': _set_export_defer_update, 'flush_update': _set_export_flush_update, 'update_tick': _set_export_update_tick, 'tag_output': _set_export_tag_output, 'scene_for_tree': _set_export_scene_for_tree, 'RenderLayerSource': _set_export_RenderLayerSource, '_key': _set_export__key, 'invalidate_render': _set_export_invalidate_render, 'capture': _set_export_capture, 'tick': _set_export_tick, 'sample': _set_export_sample, 'clear': _set_export_clear})
