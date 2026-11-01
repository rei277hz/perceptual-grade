"""Guarded native output operations, admitted by the shared exact reducer.

Native save/render handlers cannot veto Blender by raising. These operators
prepare first and invoke Blender only after consuming a validated receipt.
Computations still use lut.ensure, exactly like native compatibility handlers.
"""
import itertools
import logging

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty

from ...perceptual_grade_core.graph import scheduler_policy as policy
from ...blender_runtime.execution.diagnostics import Failure
from ...blender_runtime.ownership import OwnedField, bind_holder

LOG = logging.getLogger(__name__)
from types import SimpleNamespace
from ..integration.components import Component
from ..runtime.protocol import Protocol

def create(_application):
    _state = SimpleNamespace(_serial=itertools.count(1), _session=int(_application.app.identity, 16),
                             _active=None, _classes=[], _menus=[], _handlers=[])
    output_guard = _application.component('graph.output_guard')
    i18n = _application.component('localization.i18n')
    remove_timer = _application.app.blender.native.remove_timer


    def busy():
        return _state._active is not None


    def native_running():
        return bool(bpy.app.is_job_running('RENDER'))


    def _render_trees(scene):
        """Conservative original-ID dependency discovery, without evaluation.

        Hidden objects, unused material slots and muted node branches are included.
        They can become relevant during native evaluation; unrelated data-only
        trees are not scene dependencies. Every wrapper stays in this call.
        """
        pending = [scene]
        seen, trees = set(), set()
        supported = (bpy.types.Scene, bpy.types.Collection, bpy.types.Object,
                     bpy.types.Material, bpy.types.World, bpy.types.Light, bpy.types.NodeTree)

        def add(value):
            if isinstance(value, supported):
                pending.append(value)

        def pointer_dependencies(value):
            # Modifiers, particle settings and nodes expose original-ID references
            # through RNA. This also includes Geometry Nodes Object/Collection Info
            # and Set Material inputs, without asking a depsgraph for instances.
            for prop in value.bl_rna.properties:
                if prop.type == 'POINTER' and prop.identifier not in ('rna_type', 'pg_grade'):
                    add(getattr(value, prop.identifier, None))

        while pending:
            value = pending.pop()
            key = value.as_pointer()
            if key in seen:
                continue
            seen.add(key)
            if isinstance(value, bpy.types.Scene):
                add(value.world)
                add(value.compositing_node_group)
                add(value.background_set)
                pending.extend(value.objects)
                for layer in value.view_layers:
                    add(layer.material_override)
                editor = value.sequence_editor
                if editor is not None:
                    strips = list(editor.strips)
                    while strips:
                        strip = strips.pop()
                        add(getattr(strip, 'scene', None))
                        strips.extend(getattr(strip, 'strips', ()))
            elif isinstance(value, bpy.types.Collection):
                pending.extend(value.objects)
                pending.extend(value.children)
            elif isinstance(value, bpy.types.Object):
                add(value.instance_collection)
                add(value.data)
                for slot in value.material_slots:
                    add(slot.material)
                for modifier in value.modifiers:
                    pointer_dependencies(modifier)
                    group = getattr(modifier, 'node_group', None)
                    if group is not None:
                        for item in group.interface.items_tree:
                            if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                                add(modifier.get(item.identifier))
                for particle in value.particle_systems:
                    pointer_dependencies(particle.settings)
            elif isinstance(value, bpy.types.NodeTree):
                trees.add(key)
                for node in value.nodes:
                    pointer_dependencies(node)
                    for socket in node.inputs:
                        add(getattr(socket, 'default_value', None))
            else:
                add(value.node_tree)
        return trees


    def _nodes(scene=None):
        grades = _application.component('graph.grades')
        nodes = _application.component('graph.nodes')
        trees = None if scene is None else _render_trees(scene)
        return tuple(node for node in nodes.all_nodes() if grades.local(node.id_data)
                     and (trees is None or node.id_data.as_pointer() in trees))


    def snapshot(scene, validate=False, *, all_owners=False):
        """A complete value identity, stable across portable node replacement."""
        lut = _application.component('graph.lut')
        nodes = _application.component('graph.nodes')
        values = []
        for node in _nodes(None if all_owners else scene):
            if lut.is_adopted(node):
                # An adopted output is already the user's native scene. Its
                # model is an intent baseline, not a claim about native pixels.
                values.append((node.id_data.as_pointer(), node.name, 'native', nodes.output_intent(node)))
                continue
            model = nodes.bake_model(node)
            if validate:
                if node.get(lut.INCOMPLETE):
                    raise RuntimeError('An output is still pending installation')
                lut.validate_output(node, model, final=True)
            rows = tuple(float(value) for matrix in lut.scope_rows(node, model)
                         for row in matrix for value in row)
            values.append((node.id_data.as_pointer(), node.name, lut.cm.signature(model), rows))
        return (scene.as_pointer(), scene.frame_current, float(scene.frame_subframe),
                nodes.working_space(), tuple(sorted(values)))


    def prepare_render(scene):
        """Finish current intent through the same runtime-backed LUT barrier."""
        output_guard.require_available()
        transactions = _application.component('ownership.transactions')
        lut = _application.component('graph.lut')
        nodes = _application.component('graph.nodes')
        stacks = _application.component('graph.stacks')
        diagram = _application.component('ui.diagram')
        if nodes.recovery_pending():
            raise RuntimeError('Runtime recovery must finish before rendering')
        transactions.require_clean()
        diagram.settle_drags(commit=True)
        nodes._rendering = True
        for node in _nodes(scene):
            if stacks.is_stack(node):
                stacks.refresh(node)
            else:
                nodes.refresh_node(node)
        required = snapshot(scene)
        for node in _nodes(scene):
            if lut.is_adopted(node):continue
            model = nodes.bake_model(node)
            lut.ensure(node, model, force=True)
            lut.validate_output(node, model, final=True)
        if snapshot(scene, validate=True) != required:
            raise RuntimeError('Grading settings changed while preparing the render')
        return required


    class Operation:
        """Foreground adapter; every native invocation requires a reducer lease."""
        scene = OwnedField('output.scene')
        timer = OwnedField('output.timer')
        window_manager = OwnedField('output.window_manager')
        def __init__(self, kind, scene):
            if _state._active is not None:
                raise RuntimeError('Another output operation is already running')
            bind_holder(self, _application.app.ownership)
            self.kind, self.scene = kind, scene
            self.original_frame = (scene.frame_current, scene.frame_subframe)
            self.original_interface_lock = scene.render.use_lock_interface
            self.required = None
            self.error = None
            self.timer = None
            self.window_manager = None
            self.closed = False
            self.native_calls = 0
            self.preparing_save = False
            self.store = _application.app.store(machine=_application.machines['exact'])
            self.event('begin', operation=next(_state._serial), session=_state._session, key=next(_state._serial))
            self.coordinator = Protocol(_application, 'exact', identity={
                'operation': self.state.operation, 'generation': 1,
                'frame': int(scene.frame_current)})
            self._native_receipts = None
            _state._active = self

        @property
        def state(self):
            return self.store.state

        def event(self, kind, **values):
            state = self.state
            data = dict(operation=state.operation, session=state.session,
                        revision=state.revision, frame=state.frame)
            data.update(values)
            return self.store.dispatch(policy.ExactEvent(kind, **data))

        def prepare(self, frame=None):
            output_guard.reconcile()
            output_guard.require_available()
            transactions = _application.component('ownership.transactions')
            if transactions.pending():
                transactions.repair_all()
            transactions.require_clean()
            nodes = _application.component('graph.nodes')
            if self.state.phase != policy.EXACT_PREPARING:
                raise RuntimeError('Output preparation belongs to an inactive operation')
            target_frame = int(self.scene.frame_current if frame is None else frame)
            if self.coordinator.state.phase == 'idle':
                self.coordinator.identity['frame'] = target_frame
                self.coordinator.require('begin', 'prepare')
            elif self.coordinator.identity['frame'] != target_frame:
                self.coordinator.require('changed', 'prepare', next_frame=target_frame)
                self.coordinator.identity['frame'] = target_frame
            try:
                if frame is not None:
                    self.scene.frame_set(frame, subframe=0.)
                if self.kind == 'save':
                    self.preparing_save = True
                    try:
                        nodes.save_handler()
                    finally:
                        self.preparing_save = False
                    self.required = snapshot(self.scene, validate=True, all_owners=True)
                else:
                    self.required = prepare_render(self.scene)
                if not self.event('validated', key=self.state.required, resolution=65).accept:
                    raise RuntimeError('Output preparation was invalidated')
                self.coordinator.require('prepared', 'invoke', validated=True)
            except output_guard.PublicationDeferred:
                # A WM job may have started during preparation's explicit frame
                # evaluation. Keep PREPARING and retry from the next modal event.
                raise
            except BaseException as exc:
                self.error = Failure.capture(exc)
                self.event('fail')
                self.coordinator.event('failed')
                raise

        def authorize(self):
            output_guard.reconcile()
            output_guard.require_available()
            if self.closed or self.state.session != _state._session:
                raise RuntimeError('The output operation belongs to an expired session')
            if snapshot(self.scene, validate=True, all_owners=self.kind == 'save') != self.required:
                self.event('changed', key=next(_state._serial))
                frame = int(self.scene.frame_current)
                self.coordinator.require('changed', 'prepare', next_frame=frame)
                self.coordinator.identity['frame'] = frame
                raise RuntimeError('Grading settings changed after output preparation')
            token = next(_state._serial)
            if not self.event('native_begin', key=self.state.required, token=token).invoke:
                raise RuntimeError('Native output requires a current final-output receipt')
            if not self.coordinator.event('started').accepted:
                raise RuntimeError('Native output workflow rejected invocation')
            self._native_receipts = {outcome: self.coordinator.receipt('native_done', outcome=label)
                                     for outcome, label in ((1, 'success'), (2, 'failed'), (3, 'cancelled'))}
            self.native_calls += 1
            if self.kind != 'save':
                self.scene.render.use_lock_interface = True
            return token

        def native_result(self, outcome, token=None):
            effect = self.event('native_result', token=self.state.lease if token is None else token, outcome=outcome)
            if effect.accept and self._native_receipts is not None:
                self.coordinator.dispatch(self._native_receipts[outcome])

        def invoke(self, native):
            token = self.authorize()
            try:
                with output_guard.native_operation():
                    result = native()
            except BaseException as exc:
                self.error = Failure.capture(exc)
                self.native_result(2, token)
                raise
            if 'CANCELLED' in result:
                self.native_result(3, token)
            elif 'FINISHED' in result:
                self.native_result(1, token)
            elif 'RUNNING_MODAL' not in result:
                self.error = Failure.capture(RuntimeError('Native output returned no completion status'))
                self.native_result(2, token)
            return result

        def settle(self):
            """An asynchronous return/callback alone cannot retire the lease."""
            output_guard.reconcile()
            if self.state.phase != policy.EXACT_SETTLING:
                return False
            if self.kind != 'save' and native_running():
                return False
            token = self.state.lease
            self.event('native_stopped', token=token)
            try:
                grades = _application.component('graph.grades')
                nodes = _application.component('graph.nodes')
                if self.kind == 'save' and grades._saving:
                    nodes.save_finished_handler()
            except Exception as exc:
                self.error = Failure.capture(exc)
                self.event('cleanup_failed', token=token)
                self.coordinator.event('failed')
            self.event('settled', token=token)
            self.coordinator.event('stopped', validated=True)
            return True

        def next_frame(self, frame=None):
            if not self.event('next_frame', key=next(_state._serial)).advance:
                raise RuntimeError('An incomplete or cancelled frame cannot advance')
            frame = int(self.scene.frame_current if frame is None else frame)
            self.coordinator.require('next_frame', 'prepare', next_frame=frame)
            self.coordinator.identity['frame'] = frame

        def cancel(self):
            self.event('cancel')
            self.coordinator.event('cancel')

        def close(self):
            if self.closed:
                return True
            if self.kind != 'save' and self.state.lease and native_running():
                self.cancel()
                return False
            grades = _application.component('graph.grades')
            nodes = _application.component('graph.nodes')
            try:
                if self.state.phase == policy.EXACT_NATIVE:
                    self.native_result(2 if self.error else 3)
                self.settle()
                if self.kind == 'save':
                    if grades._saving:
                        nodes.save_finished_handler()
                else:
                    nodes.render_end_handler()
                    self.scene.render.use_lock_interface = self.original_interface_lock
                    if (self.scene.frame_current, self.scene.frame_subframe) != self.original_frame:
                        self.scene.frame_set(self.original_frame[0], subframe=self.original_frame[1])
                self.coordinator.event('close', validated=True)
                if self.coordinator.state.phase == 'settling' and not native_running():
                    self.coordinator.event('stopped', validated=True)
                if self.coordinator.state.phase != 'closed':
                    raise RuntimeError('Output workflow still owns native cleanup')
                self.coordinator.close()
            finally:
                window_manager, timer = self.window_manager, self.timer
                if timer is not None and window_manager is not None:
                    try:
                        remove_timer(window_manager, timer)
                    except (ReferenceError, RuntimeError):
                        pass
                self.timer = None
                self.window_manager = None
                self.scene = None
                self.closed = True
                if _state._active is self:
                    _state._active = None
            return True


    def reuse_save_pre():
        """Recognize portable preparation owned by a guarded native Save."""
        operation = _state._active
        if operation is None or operation.kind != 'save' or operation.state.phase != policy.EXACT_NATIVE:
            return False
        grades = _application.component('graph.grades')
        if not grades._saving or snapshot(operation.scene, validate=True, all_owners=True) != operation.required:
            operation.error = Failure.capture(RuntimeError('Prepared save output changed before serialization'))
            raise operation.error.exception()
        return True


    def reuse_render_pre():
        operation = _state._active
        if operation is None or operation.kind == 'save' or operation.state.phase != policy.EXACT_NATIVE:
            return False
        if snapshot(operation.scene, validate=True) != operation.required:
            operation.error = Failure.capture(RuntimeError('Prepared render output changed during native startup'))
            raise operation.error.exception()
        nodes = _application.component('graph.nodes')
        nodes._rendering = True
        return True


    def suspend():
        """A guarded save deliberately enters the existing suspension protocol."""
        if _state._active is None or not _state._active.preparing_save:
            return clear()
        return True


    def clear():
        _state._session += 1
        operation = _state._active
        if operation is None:
            return True
        operation.cancel()
        return operation.close()


    @persistent
    def render_complete(scene, *_args):
        operation = _state._active
        if operation is not None and operation.kind != 'save' and scene == operation.scene:
            operation.native_result(2 if operation.error else 1)


    @persistent
    def render_cancel(scene, *_args):
        operation = _state._active
        if operation is not None and operation.kind != 'save' and scene == operation.scene:
            operation.native_result(3)


    def _native_save(filepath=None, *, copy=False, compress=False):
        if filepath is None:
            if not bpy.data.filepath:
                raise RuntimeError('Choose a destination using Save As with Exact LUTs')
            return bpy.ops.wm.save_mainfile(compress=compress)
        return bpy.ops.wm.save_as_mainfile(
            filepath=filepath, check_existing=False, copy=copy, compress=compress)


    def guarded_save(filepath=None, *, copy=False, compress=False):
        output_guard.require_available()
        operation = Operation('save', bpy.context.scene)
        try:
            operation.prepare()
            result = operation.invoke(lambda: _native_save(filepath, copy=copy, compress=compress))
            operation.settle()
            if operation.error:
                raise operation.error.exception()
            return result
        finally:
            operation.close()


    def _frames(scene, animation):
        if animation and scene.render.is_movie_format:
            raise RuntimeError('Guarded animation supports image sequences; use native Render Animation for movies')
        return tuple(range(scene.frame_start, scene.frame_end + 1, scene.frame_step)) if animation else (None,)


    def _native_render(scene, frame, invocation='EXEC_DEFAULT', write_still=False):
        values = dict(scene=scene.name, animation=frame is not None)
        if frame is None:
            values['write_still'] = write_still
        else:
            values.update(frame_start=frame, frame_end=frame)
        return bpy.ops.render.render(invocation, **values)


    def guarded_render(scene=None, *, animation=False, write_still=False):
        """Synchronous/script entry point; the interactive operator shares it."""
        output_guard.require_available()
        scene = scene or bpy.context.scene
        frames = _frames(scene, animation)
        operation = Operation('animation' if animation else 'render', scene)
        try:
            for index, frame in enumerate(frames):
                if index:
                    operation.next_frame(frame)
                operation.prepare(frame)
                result = operation.invoke(lambda: _native_render(scene, frame, write_still=write_still))
                if not operation.settle():
                    raise RuntimeError('Synchronous native render did not stop')
                if operation.error:
                    raise operation.error.exception()
                if operation.state.phase != policy.EXACT_COMPLETE:
                    return {'CANCELLED'}
            return {'FINISHED'}
        finally:
            operation.close()


    class GuardedSave:
        """A blocked interactive save yields to Blender's native job cleanup."""

        def deferred_save(self, context, filepath=None, *, copy=False, compress=False):
            self._save_path, self._save_copy, self._save_compress = filepath, copy, compress
            self._operation = Operation('save', context.scene)
            try:
                self._operation.window_manager = context.window_manager
                self._operation.timer = context.window_manager.event_timer_add(.05, window=context.window)
                context.window_manager.modal_handler_add(self)
            except BaseException:
                self._operation.close()
                raise
            return {'RUNNING_MODAL'}

        def modal(self, context, event):
            operation = self._operation
            if event.type == 'ESC':
                operation.cancel()
                operation.close()
                return {'CANCELLED'}
            if event.type != 'TIMER':
                return {'PASS_THROUGH'}
            output_guard.reconcile()
            if not output_guard.available():
                return {'PASS_THROUGH'}
            try:
                if operation.state.phase == policy.EXACT_PREPARING:
                    operation.prepare()
                operation.invoke(lambda: _native_save(
                    self._save_path, copy=self._save_copy, compress=self._save_compress))
                operation.settle()
                if operation.error:
                    raise operation.error.exception()
                operation.close()
                return {'FINISHED'}
            except output_guard.PublicationDeferred:
                return {'PASS_THROUGH'}
            except Exception as exc:
                operation.close()
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}

        def cancel(self, context):
            if getattr(self, '_operation', None) is not None:
                self._operation.cancel()
                self._operation.close()


    class PG_OT_guarded_save(GuardedSave, bpy.types.Operator):
        bl_idname = 'perceptual_grade.save_guarded'
        bl_label = 'Save with Exact LUTs'
        bl_description = 'Prepare and validate final grading outputs before saving'

        @classmethod
        def poll(cls, context):
            return not busy()

        def invoke(self, context, event):
            if not bpy.data.filepath:
                result = bpy.ops.perceptual_grade.save_as_guarded('INVOKE_DEFAULT')
                # The delegated file selector owns its modal lifetime.
                return {'CANCELLED'} if 'CANCELLED' in result else {'FINISHED'}
            return self.execute(context)

        def execute(self, context):
            try:
                if not bpy.app.background and not output_guard.available():
                    return self.deferred_save(context)
                return guarded_save()
            except Exception as exc:
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}


    class PG_OT_guarded_save_as(GuardedSave, bpy.types.Operator):
        bl_idname = 'perceptual_grade.save_as_guarded'
        bl_label = 'Save As with Exact LUTs'
        bl_description = 'Choose a project file, then validate final grading outputs before saving'
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default='*.blend', options={'HIDDEN'})
        compress: BoolProperty(name='Compress', default=False)
        copy: BoolProperty(name='Save Copy', default=False)
        check_existing: BoolProperty(default=True, options={'HIDDEN'})

        @classmethod
        def poll(cls, context):
            return not busy()

        def invoke(self, context, event):
            self.filepath = bpy.data.filepath or 'Untitled.blend'
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def check(self, context):
            path = bpy.path.ensure_ext(self.filepath, '.blend')
            changed = path != self.filepath
            self.filepath = path
            return changed

        def execute(self, context):
            try:
                if not self.filepath:
                    raise RuntimeError('A destination project file is required')
                if not bpy.app.background and not output_guard.available():
                    return self.deferred_save(context, bpy.path.ensure_ext(self.filepath, '.blend'),
                                              copy=self.copy, compress=self.compress)
                return guarded_save(bpy.path.ensure_ext(self.filepath, '.blend'),
                                    copy=self.copy, compress=self.compress)
            except Exception as exc:
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}


    class GuardedRender:
        animation = False

        @classmethod
        def poll(cls, context):
            return context.scene is not None and not busy()

        def execute(self, context):
            try:
                return guarded_render(context.scene, animation=self.animation)
            except Exception as exc:
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}

        def invoke(self, context, event):
            if bpy.app.background:
                return self.execute(context)
            try:
                self._frames = _frames(context.scene, self.animation)
                self._index = 0
                self._operation = Operation('animation' if self.animation else 'render', context.scene)
                self._operation.window_manager = context.window_manager
                self._operation.timer = context.window_manager.event_timer_add(.05, window=context.window)
                context.window_manager.modal_handler_add(self)
                return {'RUNNING_MODAL'}
            except Exception as exc:
                if getattr(self, '_operation', None) is not None:
                    self._operation.close()
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}

        def modal(self, context, event):
            operation = self._operation
            if event.type == 'ESC':
                operation.cancel()
                if native_running():
                    return {'PASS_THROUGH'}
            if event.type != 'TIMER' and not operation.state.cancelled:
                return {'PASS_THROUGH'}
            output_guard.reconcile()
            try:
                if operation.closed:
                    return {'CANCELLED'}
                if operation.state.cancelled and not native_running():
                    operation.close()
                    return {'CANCELLED'}
                if operation.state.phase == policy.EXACT_PREPARING:
                    if not output_guard.available():
                        return {'PASS_THROUGH'}
                    frame = self._frames[self._index]
                    operation.prepare(frame)
                if operation.state.phase == policy.EXACT_READY:
                    if not output_guard.available():
                        return {'PASS_THROUGH'}
                    frame = self._frames[self._index]
                    operation.invoke(lambda: _native_render(operation.scene, frame, 'INVOKE_DEFAULT'))
                if operation.settle():
                    if operation.error:
                        raise operation.error.exception()
                    if operation.state.phase != policy.EXACT_COMPLETE:
                        operation.close()
                        return {'CANCELLED'}
                    self._index += 1
                    if self._index == len(self._frames):
                        operation.close()
                        return {'FINISHED'}
                    operation.next_frame(self._frames[self._index])
                return {'PASS_THROUGH'}
            except output_guard.PublicationDeferred:
                return {'PASS_THROUGH'}
            except Exception as exc:
                operation.error = Failure.capture(exc)
                operation.cancel()
                operation.close()
                _application.component('integration.outcomes').failure(None, 'output', exc)
                return {'CANCELLED'}

        def cancel(self, context):
            if getattr(self, '_operation', None) is not None:
                self._operation.cancel()
                self._operation.close()


    class PG_OT_guarded_render(GuardedRender, bpy.types.Operator):
        bl_idname = 'perceptual_grade.render_guarded'
        bl_label = 'Render with Exact LUTs'
        bl_description = 'Validate final grading outputs before starting a render'


    class PG_OT_guarded_animation(GuardedRender, bpy.types.Operator):
        bl_idname = 'perceptual_grade.render_animation_guarded'
        bl_label = 'Render Image Sequence with Exact LUTs'
        bl_description = 'Validate each frame before its native image-sequence render'
        animation = True


    def file_menu(self, context):
        self.layout.separator()
        self.layout.operator(PG_OT_guarded_save.bl_idname)
        self.layout.operator(PG_OT_guarded_save_as.bl_idname)


    def render_menu(self, context):
        self.layout.separator()
        self.layout.operator(PG_OT_guarded_render.bl_idname)
        self.layout.operator(PG_OT_guarded_animation.bl_idname)


    file_menu = _application.component('integration.outcomes').callback(file_menu, control='save.menu')
    render_menu = _application.component('integration.outcomes').callback(render_menu, control='render.menu')

    def register():
        for cls in (PG_OT_guarded_save, PG_OT_guarded_save_as, PG_OT_guarded_render, PG_OT_guarded_animation):
            _application.register_class(cls)
            _state._classes.append(cls)
        for menu, callback in ((bpy.types.TOPBAR_MT_file, file_menu), (bpy.types.TOPBAR_MT_render, render_menu)):
            menu.append(callback)
            _state._menus.append((menu, callback))
        for name, callback in (('render_complete', render_complete), ('render_cancel', render_cancel)):
            native = _application.add_handler(name, callback)
            if (name, native) not in _state._handlers:
                _state._handlers.append((name, native))


    def unregister():
        # Lifecycle keeps this registration snapshot and resumes teardown after
        # Blender releases the active native job. Its callback may still arrive.
        if not clear():
            return False
        errors, handlers, menus, classes = [], [], [], []
        for name, callback in _state._handlers:
            try:
                _application.remove_handler(name, callback)
            except Exception as error:
                errors.append(error)
                handlers.append((name, callback))
        _state._handlers[:] = handlers
        for menu, callback in _state._menus:
            try:
                if callback in getattr(menu.draw, '_draw_funcs', ()):
                    menu.remove(callback)
            except Exception as error:
                errors.append(error)
                menus.append((menu, callback))
        _state._menus[:] = menus
        for cls in reversed(_state._classes):
            try:
                _application.unregister_class(cls)
            except Exception as error:
                errors.append(error)
                classes.append(cls)
        _state._classes[:] = reversed(classes)
        if errors:
            raise ExceptionGroup('Guarded output registration cleanup failed', errors)
        return True

    def _set_export_busy(value):
        nonlocal busy
        busy = value

    def _set_export_native_running(value):
        nonlocal native_running
        native_running = value

    def _set_export__render_trees(value):
        nonlocal _render_trees
        _render_trees = value

    def _set_export__nodes(value):
        nonlocal _nodes
        _nodes = value

    def _set_export_snapshot(value):
        nonlocal snapshot
        snapshot = value

    def _set_export_prepare_render(value):
        nonlocal prepare_render
        prepare_render = value

    def _set_export_Operation(value):
        nonlocal Operation
        Operation = value

    def _set_export_reuse_save_pre(value):
        nonlocal reuse_save_pre
        reuse_save_pre = value

    def _set_export_reuse_render_pre(value):
        nonlocal reuse_render_pre
        reuse_render_pre = value

    def _set_export_suspend(value):
        nonlocal suspend
        suspend = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_render_complete(value):
        nonlocal render_complete
        render_complete = value

    def _set_export_render_cancel(value):
        nonlocal render_cancel
        render_cancel = value

    def _set_export__native_save(value):
        nonlocal _native_save
        _native_save = value

    def _set_export_guarded_save(value):
        nonlocal guarded_save
        guarded_save = value

    def _set_export__frames(value):
        nonlocal _frames
        _frames = value

    def _set_export__native_render(value):
        nonlocal _native_render
        _native_render = value

    def _set_export_guarded_render(value):
        nonlocal guarded_render
        guarded_render = value

    def _set_export_GuardedSave(value):
        nonlocal GuardedSave
        GuardedSave = value

    def _set_export_PG_OT_guarded_save(value):
        nonlocal PG_OT_guarded_save
        PG_OT_guarded_save = value

    def _set_export_PG_OT_guarded_save_as(value):
        nonlocal PG_OT_guarded_save_as
        PG_OT_guarded_save_as = value

    def _set_export_GuardedRender(value):
        nonlocal GuardedRender
        GuardedRender = value

    def _set_export_PG_OT_guarded_render(value):
        nonlocal PG_OT_guarded_render
        PG_OT_guarded_render = value

    def _set_export_PG_OT_guarded_animation(value):
        nonlocal PG_OT_guarded_animation
        PG_OT_guarded_animation = value

    def _set_export_file_menu(value):
        nonlocal file_menu
        file_menu = value

    def _set_export_render_menu(value):
        nonlocal render_menu
        render_menu = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component({
        'busy': busy,
        'native_running': native_running,
        '_render_trees': _render_trees,
        '_nodes': _nodes,
        'snapshot': snapshot,
        'prepare_render': prepare_render,
        'Operation': Operation,
        'reuse_save_pre': reuse_save_pre,
        'reuse_render_pre': reuse_render_pre,
        'suspend': suspend,
        'clear': clear,
        'render_complete': render_complete,
        'render_cancel': render_cancel,
        '_native_save': _native_save,
        'guarded_save': guarded_save,
        '_frames': _frames,
        '_native_render': _native_render,
        'guarded_render': guarded_render,
        'GuardedSave': GuardedSave,
        'PG_OT_guarded_save': PG_OT_guarded_save,
        'PG_OT_guarded_save_as': PG_OT_guarded_save_as,
        'GuardedRender': GuardedRender,
        'PG_OT_guarded_render': PG_OT_guarded_render,
        'PG_OT_guarded_animation': PG_OT_guarded_animation,
        'file_menu': file_menu,
        'render_menu': render_menu,
        'register': register,
        'unregister': unregister,
    }, state=_state, writers={'busy': _set_export_busy, 'native_running': _set_export_native_running, '_render_trees': _set_export__render_trees, '_nodes': _set_export__nodes, 'snapshot': _set_export_snapshot, 'prepare_render': _set_export_prepare_render, 'Operation': _set_export_Operation, 'reuse_save_pre': _set_export_reuse_save_pre, 'reuse_render_pre': _set_export_reuse_render_pre, 'suspend': _set_export_suspend, 'clear': _set_export_clear, 'render_complete': _set_export_render_complete, 'render_cancel': _set_export_render_cancel, '_native_save': _set_export__native_save, 'guarded_save': _set_export_guarded_save, '_frames': _set_export__frames, '_native_render': _set_export__native_render, 'guarded_render': _set_export_guarded_render, 'GuardedSave': _set_export_GuardedSave, 'PG_OT_guarded_save': _set_export_PG_OT_guarded_save, 'PG_OT_guarded_save_as': _set_export_PG_OT_guarded_save_as, 'GuardedRender': _set_export_GuardedRender, 'PG_OT_guarded_render': _set_export_PG_OT_guarded_render, 'PG_OT_guarded_animation': _set_export_PG_OT_guarded_animation, 'file_menu': _set_export_file_menu, 'render_menu': _set_export_render_menu, 'register': _set_export_register, 'unregister': _set_export_unregister})
