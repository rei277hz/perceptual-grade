"""Native Shader/Compositor group construction and persistent instance controls."""

from ..integration.components import Component

def create(_application):
    native = _application.component('localization.i18n').native
    from ...perceptual_grade_core.localization.controls import control_tooltip
    tr = _application.component('localization.i18n').tr; CONTEXT = _application.component('localization.i18n').CONTEXT

    import logging
    from ...blender_runtime.execution.diagnostics import log_exception
    import bpy
    from bpy.app.handlers import persistent
    from bpy.props import FloatProperty, IntProperty, EnumProperty, PointerProperty
    from ...perceptual_grade_core.model import linear as cm, tone as tm, color as ch
    cube_nodes = _application.component('graph.lut'); grades = _application.component('graph.grades')
    from ...perceptual_grade_core.model import transforms
    controls = _application.component('model.controls')
    spaces = _application.component('graph.spaces')
    editing = _application.component('graph.editing')
    ownership = _application.component('ownership')

    LOG = logging.getLogger(__package__)

    def _lifecycle_event(kind):
        scheduler = _application.component('graph.scheduler')
        return scheduler.lifecycle_event(kind)


    VERSION = 3
    RELEASE = '1.21.0'
    TAG = '_perceptual_grade'
    KERNEL = '_perceptual_grade_kernel'
    _updating = set()
    _signatures = {}
    _intent_revisions = {}
    _intent_epoch = None
    _intent_serial = 0
    _registered = False
    _class_registered = False
    _rendering = False
    _replacement_pending = False
    _recovery = None
    _recovery_work = None
    _recovery_error = ''


    def recovery_pending():
        """The lifecycle reducer fences automatic callbacks during RNA recovery."""
        scheduler = _application.component('graph.scheduler')
        protocol = _application.app.lifecycle
        return (not scheduler.lifecycle_admits('discover') or
                protocol is not None and protocol.state.phase != 'active')


    def working_space():
        info = bpy.data.colorspace
        interop = getattr(info, 'working_space_interop_id', '')
        name = cm.INTEROP.get(interop, info.working_space)
        if name not in cm.SPACES:
            raise ValueError(tr('error.working_space',space=name))
        return name


    def output_intent(node):
        """Semantic controls only; derived native rows/banks are never authority.

        A restored selector, external graph edit, or force refresh does not
        change this identity. A Grade reference/control, Stack occurrence, or
        working-space change does authorize a new requested output.
        """
        def frozen(value):
            if isinstance(value,dict):return tuple((key,frozen(item)) for key,item in sorted(value.items()))
            if isinstance(value,(tuple,list)):return tuple(frozen(item) for item in value)
            return value
        def source(value):
            if value is None:return None
            owner=grades.data_owner(value)
            identity=int(getattr(owner,'session_uid',owner.id_data.session_uid))
            return identity,frozen(grades.snapshot(owner))
        info=bpy.data.colorspace
        context=(str(info.working_space),str(getattr(info,'working_space_interop_id','')))
        stacks=_application.component('graph.stacks')
        controls=(tuple((item.key,bool(item.enabled),item.direction,source(item.grade))
                        for item in node.pg_stack_entries) if stacks.is_stack(node) else source(node))
        return ('native-output-intent-v1',context,'stack' if stacks.is_stack(node) else 'grade',controls)


    def is_grade(node):
        return bool(node and node.get(TAG) == VERSION
                    and node.get(grades.NODE_SCHEMA) == grades.SCHEMA
                    and getattr(node, 'node_tree', None)
                    and node.node_tree.get(KERNEL) == VERSION
                    and node.node_tree.get('_pg_cube_schema') == cube_nodes.SCHEMA
                    and _application.component('graph.output_resources').is_current(node))


    def observe_intent(node, intent=None):
        """Observe this source's semantic edits without notifying its consumers."""
        nonlocal _intent_epoch, _intent_serial
        epoch = ownership.generation_key()
        if _intent_epoch != epoch:
            _intent_revisions.clear()
            _intent_epoch = epoch
        key = (int(node.id_data.session_uid), node.as_pointer())
        intent = output_intent(node) if intent is None else intent
        previous = _intent_revisions.get(key)
        if previous is None or previous[0] != intent:
            _intent_serial += 1
            _intent_revisions[key] = (intent, _intent_serial)
        return _intent_revisions[key][1]


    def resource_action(node):
        """Discover structural work without performing it in a timer or handler."""
        stacks = _application.component('graph.stacks')
        if node is None or node.id_data.is_evaluated or not grades.local(node.id_data):
            return None
        if node.bl_idname not in (*grades.KINDS.values(), *stacks.KINDS.values()) and not (
                is_grade(node) or stacks.is_stack(node)):
            return None
        kernel = getattr(node, 'node_tree', None)
        if node.get('_pg_pending_init') and kernel is None:
            return 'initialize'
        if kernel is None or kernel.get('_pg_cube_schema') != cube_nodes.SCHEMA:
            return 'invalid'
        output_resources = _application.component('graph.output_resources')
        if not output_resources.is_current(node) or any(
                item.get('_pg_cube_address') and item.node_tree is None for item in kernel.nodes):
            return 'invalid'
        try:
            if not output_resources.writable_banks(node):
                return 'independent'
        except ValueError:
            return 'invalid'
        if (node.get('_pg_pending_copy') or not grades.local(kernel)
                or kernel.users - int(kernel.use_fake_user) > 1):
            return 'independent'
        if not _application.component('graph.bake').output_current(kernel):
            return 'invalid'
        return None


    def resources_ready(node):
        stacks = _application.component('graph.stacks')
        return (bool(is_grade(node) or stacks.is_stack(node))
                and not node.id_data.is_evaluated and grades.local(node.id_data)
                and resource_action(node) is None)


    def copy_output_resources(node, original):
        """Isolate a native individual duplicate before its operator snapshots it.

        Blender passes an ownerless original wrapper to copy(). Membership in the
        destination tree distinguishes individual duplication (whose ID users have
        already been counted) from whole-tree copying (whose users are counted only
        after this callback). Never adjust native user counts ourselves.
        """
        if node.id_data.is_evaluated:
            return
        node['_pg_pending_copy'] = True
        source = original.as_pointer()
        if not any(item.as_pointer() == source for item in node.id_data.nodes):
            return
        kernel = getattr(original, 'node_tree', None)
        if not (is_grade(original) or _application.component('graph.stacks').is_stack(original)):
            return
        node.node_tree = cube_nodes.clone_kernel(kernel)
        del node['_pg_pending_copy']


    def prepare_output_resources(node):
        """Complete structural work from an explicit undoable operation."""
        stacks = _application.component('graph.stacks')
        action = resource_action(node)
        if action is None:
            return
        if node.id_data.is_evaluated or not grades.local(node.id_data):
            raise ValueError('Cannot initialize borrowed or linked output resources')
        if action == 'initialize':
            if node.bl_idname in stacks.KINDS.values():stacks.initialize(node)
            else:initialize_grade(node)
            return
        if action != 'independent':
            raise ValueError('Unsupported or incomplete output resources')
        kernel = node.node_tree
        replacement = cube_nodes.clone_kernel(kernel)
        assign_kernel(node, replacement)
        for key in ('_pg_pending_init', '_pg_pending_copy'):
            if key in node:del node[key]
        _signatures.pop(node.as_pointer(), None)
        if stacks.is_stack(node):
            stacks.bind_dependencies(node)
            stacks.refresh(node, True)
        else:
            grades.bind_animation_dependency(node)
            refresh_node(node, True)


    def active_grade(context):
        if recovery_pending():
            return None
        space = context.space_data
        if not space or space.type != 'NODE_EDITOR':
            return None
        tree = space.edit_tree
        node = tree.nodes.active if tree else None
        if is_grade(node):return node
        stacks = _application.component('graph.stacks')
        return editing.active(context,node) if stacks.is_stack(node) else None


    def editable(node):
        if grades.is_data(node):
            return grades.local(node) and not node.is_evaluated
        if editing.is_target(node):
            return editing.valid(node) and resources_ready(node.stack)
        return resources_ready(node) and grades.editable(node)


    def read_parameters(node):
        return grades.parameters(node)


    def grading_space(node):
        return grades.space(node)


    def linear_matrix(node):
        import numpy as np
        space = grading_space(node)
        return np.asarray(cm.rgb_matrix(grades.matrix(node), space))


    def read_tone(node):
        node = grades.data_owner(node)
        import numpy as np
        p=node.pg_tone;defaults=tm.Parameters();values={}
        for name in tm.PARAM_NAMES:
            value=int(p.peak_nits) if name=='peak_nits' else getattr(p,name)
            default=getattr(defaults,name)
            # RNA float properties store binary32. Recover the exact declared
            # default, without treating nearby edited values as a reset.
            values[name]=default if value==float(np.float32(default)) else value
        return tm.Parameters(**values)


    def read_chromatic(node):
        node = grades.data_owner(node)
        return ch.Parameters(**{n:getattr(node.pg_chromatic,n) for n in ch.PARAM_NAMES})


    def color_model(node,space=None):
        space=grading_space(node)
        return ch.model(read_tone(node),read_chromatic(node),space)


    def owner(settings):
        # Blender 5.2 cannot create path_from_id() for a PropertyGroup attached to
        # a native Node. Resolve within its owning tree by pointer instead.
        try:
            if grades.is_data(settings.id_data):
                return settings.id_data
            pointer = settings.as_pointer()
            for node in settings.id_data.nodes:
                if is_grade(node) and (node.pg_linear.as_pointer() == pointer or
                        (hasattr(node,'pg_tone') and node.pg_tone.as_pointer() == pointer) or
                        (hasattr(node,'pg_chromatic') and node.pg_chromatic.as_pointer() == pointer)):
                    return node
        except (ValueError, AttributeError, ReferenceError):
            pass
        return None


    def changed(settings, context):
        if recovery_pending():
            return
        node = owner(settings)
        if node is not None and node.as_pointer() not in _updating:
            grades.changed(node)


    def settings(value,space=None):
        return transforms.Settings(grades.matrix(value),read_tone(value),read_chromatic(value),grading_space(value))


    def bake_model(node):
        stacks = _application.component('graph.stacks')
        if stacks.is_stack(node):return stacks.model(node)
        value=settings(node)
        return transforms.Stack([transforms.Settings(cm.I3,value.tone,value.color,value.space)],'Linear Rec.709')




    class PGLinearSettings(bpy.types.PropertyGroup):
        exposure: FloatProperty(translation_context=CONTEXT, name=native('control.exposure.label'), default=0, min=-6, max=6,
            soft_min=-6, soft_max=6, step=1, precision=2, update=changed,
            description=native('tooltip.linear.exposure'))
        source_x: FloatProperty(translation_context=CONTEXT, name=native('control.source_x.label'), default=.3127, min=0., max=.9,
            step=1, precision=5,
            description=native('tooltip.linear.source_x'))
        source_y: FloatProperty(translation_context=CONTEXT, name=native('control.source_y.label'), default=.329, min=0., max=.9,
            step=1, precision=5,
            description=native('tooltip.linear.source_y'))
        cr_balance: FloatProperty(translation_context=CONTEXT, name=native('control.cr_balance.label'), default=0, min=-2, max=2,
            step=1, precision=2, update=changed)
        yb_balance: FloatProperty(translation_context=CONTEXT, name=native('control.yb_balance.label'), default=0, min=-2, max=2,
            step=1, precision=2, update=changed)
        cr_spread: FloatProperty(translation_context=CONTEXT, name=native('control.cr_spread.label'), default=0, min=-100, max=150,
            step=10, precision=1, update=changed,
            description=native('tooltip.linear.cr_spread'))
        yb_spread: FloatProperty(translation_context=CONTEXT, name=native('control.yb_spread.label'), default=0, min=-100, max=150,
            step=10, precision=1, update=changed,
            description=native('tooltip.linear.yb_spread'))
        cr_heading: FloatProperty(translation_context=CONTEXT, name=native('control.cr_heading.label'), default=0, min=-180, max=180,
            step=10, precision=1, update=changed,
            description=native('tooltip.linear.cr_heading'))
        yb_heading: FloatProperty(translation_context=CONTEXT, name=native('control.yb_heading.label'), default=0, min=-180, max=180,
            step=10, precision=1, update=changed,
            description=native('tooltip.linear.yb_heading'))


    def linear_property(name, spec):
        def get(settings):
            value = owner(settings)
            return getattr(grades.parameters(value), name) if value is not None else getattr(cm.Parameters(), name)
        def set_(settings, value):
            target = owner(settings)
            if target is not None:
                write_values(target, {name: value})
        kwargs = dict(spec.keywords)
        kwargs.pop('update', None)
        lo, hi = kwargs['min'], kwargs['max']
        kwargs.update(get=get, set=set_, options=set(), min=-1e12, max=1e12,
                      soft_min=lo, soft_max=hi)
        return FloatProperty(**kwargs)


    for _name in cm.PARAM_NAMES:
        PGLinearSettings.__annotations__[_name] = linear_property(_name, PGLinearSettings.__annotations__[_name])


    def tone_callback(field):
        def update(settings, context):
            if recovery_pending():return
            node=owner(settings)
            if node is None or node.as_pointer() in _updating or node.as_pointer() in grades._busy:return
            key=node.as_pointer();_updating.add(key)
            try:
                minimum=2*settings.reference_nits
                if int(settings.peak_nits)<minimum:
                    settings.peak_nits=str(next(p for p in tm.PEAKS if p>=minimum))
                if field in tm.LEVEL_FIELDS:
                    current=tm.Parameters(**{n:int(settings.peak_nits) if n=='peak_nits' else getattr(settings,n) for n in tm.PARAM_NAMES})
                    value=tm.clamp_landmark(current,tm.LEVEL_FIELDS.index(field),getattr(settings,field),grading_space(node))
                    if abs(value-getattr(settings,field))>1e-6:setattr(settings,field,value)
            finally:_updating.discard(key)
            grades.changed(node)
        return update


    def reference_get(settings):return 203


    def reference_set(settings,value):
        if value != 203:raise ValueError('Only cached D65 / 203 nits is supported')


    class PGToneSettings(bpy.types.PropertyGroup):
        input_black: FloatProperty(translation_context=CONTEXT, name=native('control.input_black.label'),default=0,min=0,max=2,step=1,precision=3,
            update=tone_callback('input_black'),description=native('tooltip.tone.black'))
        input_shadows: FloatProperty(translation_context=CONTEXT, name=native('control.input_shadows.label'),default=tm.LOW_Y*203,min=.01,max=50,step=1,precision=3,
            update=tone_callback('input_shadows'),description=native('tooltip.tone.shadows'))
        input_midtones: FloatProperty(translation_context=CONTEXT, name=native('control.input_midtones.label'),default=36.54,min=.1,max=250,step=10,precision=2,
            update=tone_callback('input_midtones'),description=native('tooltip.tone.midtones'))
        input_white: FloatProperty(translation_context=CONTEXT, name=native('control.input_white.label'),default=203,min=10,max=1000,step=10,precision=2,
            update=tone_callback('input_white'),description=native('tooltip.tone.white'))
        input_highlights: FloatProperty(translation_context=CONTEXT, name=native('control.input_highlights.label'),default=1000,min=100,max=10000,step=10,precision=1,
            update=tone_callback('input_highlights'),description=native('tooltip.tone.highlights'))
        fit: IntProperty(translation_context=CONTEXT, name=native('control.remap_fit.label'),default=0,min=-2,max=6,
            update=tone_callback('fit'),description=native('tooltip.tone.fit'))
        reference_nits: IntProperty(translation_context=CONTEXT, name=native('control.reference_nits.label'),default=203,min=48,max=1000,
            get=reference_get,set=reference_set,description=native('tooltip.tone.reference'))
        peak_nits: EnumProperty(translation_context=CONTEXT, name=native('control.peak_nits.label'),default='1000',items=[(str(p),native(f'peak.option.{p}'),'') for p in tm.PEAKS],
            update=tone_callback('peak_nits'))



    def focus_changed(settings,context):
        if context and context.screen:
            for area in context.screen.areas:
                if area.type=='NODE_EDITOR':area.tag_redraw()


    class PGChromaticSettings(bpy.types.PropertyGroup):
        family: EnumProperty(translation_context=CONTEXT, name=native('family.label'),items=[('J',native('family.j.label'),native('tooltip.family.j')),('C',native('family.c.label'),native('tooltip.family.c')),('WARP',native('family.warp.label'),native('tooltip.family.warp'))],default='J',update=focus_changed)
        active_handle: IntProperty(name='Selected Warp Handle',default=-1,min=-1,max=5,update=focus_changed)
        active_tone: IntProperty(translation_context=CONTEXT, name='Selected Tone',default=1,min=0,max=3,update=focus_changed)
        active_chroma: EnumProperty(translation_context=CONTEXT, items=[('low','Cₗ',''),('high','Cₕ','')],default='low',update=focus_changed)


    for name,label,default,lo,hi,precision,dctl in controls.CONTROLS:
        factory=IntProperty if precision==0 else FloatProperty
        args=dict(name=native(label),translation_context=CONTEXT,default=default,min=lo,max=hi,update=changed,
                  description=native(control_tooltip('color', name)))
        if precision:args.update(precision=precision,step=5 if name=='adaptance_half_saturation' else (10 if precision==1 else 1))
        if name.startswith('web_scale_'):args.update(soft_min=-1.,soft_max=1.,precision=2)
        PGChromaticSettings.__annotations__[name]=factory(**args)


    def write_chromatic(node,values):
        if not editable(node):return
        target=grades.data_owner(node)
        key=target.as_pointer();_updating.add(key)
        try:
            for name,value in values.items():setattr(target.pg_chromatic,name,value)
        finally:_updating.discard(key)
        grades.changed(target)


    def keyframe_chromatic(node,name):
        grades.keyframe(node, 'pg_chromatic', name, tr('animation.color'))


    def write_tone(node,values):
        if not editable(node):return
        target=grades.data_owner(node)
        key=target.as_pointer();_updating.add(key)
        try:
            for name,value in values.items():setattr(target.pg_tone,name,str(value) if name=='peak_nits' else value)
            p=target.pg_tone
            if int(p.peak_nits)<2*p.reference_nits:p.peak_nits=str(next(v for v in tm.PEAKS if v>=2*p.reference_nits))
        finally:_updating.discard(key)
        grades.changed(target)


    def keyframe_tone(node,name):
        grades.keyframe(node, 'pg_tone', name, tr('animation.tone'))


    def write_values(node, values):
        """Batch a two-coordinate drag into one matrix update."""
        if not editable(node):return
        if 'source_x' in values or 'source_y' in values:
            values = dict(values)
            previous=read_parameters(node)
            point = (values.get('source_x', previous.source_x),
                     values.get('source_y', previous.source_y))
            if 'source_x' in values and 'source_y' in values:
                point = cm.safe_source_white(point)
            else:
                point = cm.source_component(point, 0 if 'source_x' in values else 1)
            values.update(zip(('source_x', 'source_y'), point))
        grades.edit_linear(node, values)


    def keyframe_parameter(node, name):
        grades.keyframe(node, 'pg_linear', name, tr('animation.linear'))


    def reset_linear(node):
        if grades.matrix(node) == cm.I3:return
        grade=grades.ensure(node)
        grades.write_matrix(grade,cm.I3)
        grades.publish(grade)


    def refresh_node(node, force=False, immediate=False):
        if recovery_pending() or not is_grade(node) or not grades.local(node.id_data):
            return
        observe_intent(node)
        if not resources_ready(node):
            return
        if not cube_nodes.request_intent(node,output_intent(node)):
            return
        grades.sync_node(node)
        i18n = _application.component('localization.i18n')
        node_presentation = _application.component('graph.presentation')
        node_presentation.node_options(node)
        i18n.update_node_title(node)
        key = node.as_pointer()
        if key in _updating:
            return
        compositor = _application.component('graph.compositor')
        jobs = _application.component('scopes.jobs')
        jobs.refresh_linear(node)
        # Baking observes edits immediately, even when native rows must wait for
        # an existing compositor frame. Stage 1 is excluded from this bake key.
        model=bake_model(node)
        cube_nodes.request(node,model,force=_rendering)
        if not immediate and not _rendering and compositor.defer_update(node,force):return
        _updating.add(key)
        try:
            signal_before=cube_nodes.signal_signature(node)
            comparison = _application.component('scopes.reference')
            comparison.sync_connection(node)
            try:
                space = working_space()
                grading = grading_space(node)
                tone=read_tone(node);chromatic=read_chromatic(node)
                sig = (space, grading, grades.matrix(node),tone,chromatic)
                if not force and _signatures.get(key) == sig:
                    return
                before, after = spaces.adapters(space)
                lut_before = cube_nodes.cm.TO_LUT @ linear_matrix(node) @ before
                lut_after = after @ cube_nodes.cm.FROM_LUT
                error = ''
            except (ValueError, ArithmeticError, AttributeError) as exc:
                sig, error = ('error', str(exc)), str(exc)
                space = 'Unsupported'
                if not force and _signatures.get(key) == sig:
                    return
            if not error:
                if not spaces.uniforms(node,lut_before,lut_after):return
                model=bake_model(node)
                cube_nodes.request(node,model,force=_rendering)
            _signatures[key] = sig
            if node.get('_pg_error', '') != error:
                node['_pg_error'] = error
            if node.get('_pg_space') != space:
                node['_pg_space'] = space
            if force or error or cube_nodes.signal_signature(node)!=signal_before:
                compositor.tag_output(node)
        finally:
            _updating.discard(key)


    def set_uniform(socket,value):
        if socket is None:return
        f32 = _application.component('graph.optimize').f32
        value=tuple(f32(v) for v in value) if hasattr(value,'__len__') else f32(value)
        old=tuple(socket.default_value) if hasattr(socket.default_value,'__len__') else socket.default_value
        if old!=value:socket.default_value=value


    def all_grades():
        yield from all_nodes(False)


    def all_nodes(include_stacks=True):
        stacks = _application.component('graph.stacks')
        seen = set()
        trees = list(bpy.data.node_groups)
        trees += [m.node_tree for m in bpy.data.materials if m.node_tree]
        trees += [w.node_tree for w in bpy.data.worlds if w.node_tree]
        trees += [light.node_tree for light in bpy.data.lights if light.node_tree]
        for tree in trees:
            key = tree.as_pointer()
            if key in seen:
                continue
            seen.add(key)
            for node in tree.nodes:
                if is_grade(node) or (include_stacks and stacks.is_stack(node)):
                    yield node


    def refresh_all(force=False):
        if recovery_pending() or not _registered or not hasattr(bpy.data, 'node_groups') or grades._converting or grades._saving:
            return
        compositor = _application.component('graph.compositor')
        if compositor._busy:return
        node_presentation = _application.component('graph.presentation')
        grades.representations()
        stacks = _application.component('graph.stacks')
        live=tuple(all_nodes())
        scheduler = _application.component('graph.scheduler')
        scheduler.prune(live)
        for node in live:
            try:
                if stacks.is_stack(node):stacks.refresh(node,force)
                else:refresh_node(node,force)
            except (ReferenceError, RuntimeError, ValueError, TypeError) as exc:
                log_exception(LOG, 'Could not update Perceptual Grade node')


    @persistent
    def sync_handler(*_args):
        # Native depsgraph/frame callbacks may run from a render thread or while
        # evaluation is already in progress. The main-thread poll owns refresh.
        # In particular, a synchronous cache hit must not publish from here.
        capture = _application.components.get('scopes.capture')
        if capture is not None:
            capture.invalidate()
        return None


    @persistent
    def render_init_handler(scene,*_args):
        """A render lease spans every frame through complete/cancel."""
        output_guard = _application.component('graph.output_guard')
        with output_guard.native_callback():
            output_guard.evaluation_started('render',scene.session_uid)


    @persistent
    def render_complete_handler(scene,*_args):
        output_guard = _application.component('graph.output_guard')
        with output_guard.native_callback():
            render_end_handler()
            output_guard.evaluation_finished('render',scene.session_uid)


    @persistent
    def render_handler(*_args):
        """Validate pre-existing output; native handlers cannot veto rendering."""
        output = _application.component('graph.output'); output_guard = _application.component('graph.output_guard')
        nonlocal _rendering
        with output_guard.native_callback():
            if output.reuse_render_pre():return
            _rendering=True
            if recovery_pending():
                LOG.warning('Perceptual Grade output recovery is pending; use Render with Exact LUTs')
                return
            try:
                output.snapshot(bpy.context.scene, validate=True)
            except (ValueError, RuntimeError, ArithmeticError, ReferenceError):
                # A Python exception cannot stop native rendering. Leave all
                # native payloads and selectors alone and report this limitation.
                log_exception(LOG, 'Perceptual Grade retains its previous output: current exact output '
                              'is pending; use Render with Exact LUTs')


    @persistent
    def render_end_handler(*_args):
        # A native post/cancel callback is still inside the render operation.
        # The regular application timer will request refreshed output afterward.
        nonlocal _rendering
        _rendering=False


    @persistent
    def scope_render_handler(scene,*_args):
        if recovery_pending():
            return
        compositor = _application.component('graph.compositor')
        scopes = _application.component('scopes.sampling')
        compositor.invalidate_render(scene)
        capture = _application.components.get('scopes.capture')
        if capture is not None:capture.invalidate()


    def _recover_steps(actions, strict=False):
        """Attempt every independent cleanup even when one adapter fails."""
        from ...blender_runtime.execution.diagnostics import Failure
        errors=[]
        for action in actions:
            try:
                action()
            except Exception as exc:
                errors.append(Failure.capture(exc))
                log_exception(LOG, 'Perceptual Grade recovery step failed: %s', getattr(action, '__name__', action))
        if strict and errors:raise errors[0].exception('Perceptual Grade recovery cleanup failed')


    @persistent
    def suspend_handler(*_args):
        """Fence callbacks before load/undo/redo replace Blender's RNA objects."""
        output = _application.component('graph.output')
        output.suspend()
        nonlocal _rendering
        scheduler = _application.component('graph.scheduler')
        diagram = _application.component('ui.diagram')
        _rendering = False
        try:
            state = _lifecycle_event('quiesce')
        except RuntimeError:
            state = _lifecycle_event('recover')
        scheduler.recover()
        _recover_steps((diagram.cleanup_drag_timers, diagram.cleanup_hover_cursors,
                        diagram._routers.clear, grades.recover_flags, _updating.clear), strict=state.phase == 3)
        return state.phase == 3


    @persistent
    def save_handler(*_args):
        output = _application.component('graph.output'); output_guard = _application.component('graph.output_guard'); output_resources = _application.component('graph.output_resources')
        transactions = _application.component('ownership.transactions')
        if output.reuse_save_pre():
            transactions.require_clean()
            return
        output_guard.require_available()
        diagram = _application.component('ui.diagram')
        scheduler = _application.component('graph.scheduler')
        if recovery_pending():
            raise RuntimeError('Runtime recovery must finish before saving')
        try:
            transactions.require_clean()
            # A save is a real edit boundary: accept the most recent buffered pair
            # and auto-key it before replacing nodes or serializing the project.
            diagram.settle_drags(commit=True)
            recovering = suspend_handler()
            refresh_all()
            if not recovering:
                _lifecycle_event('save_begin')
            for node in all_nodes():
                if grades.local(node.id_data):
                    if not cube_nodes.request_intent(node,output_intent(node)):
                        continue
                    model = bake_model(node)
                    cube_nodes.ensure(node, model, force=True)
                    cube_nodes.validate_output(node, model, final=True)
                    output_resources.pack_output(node)
            if recovering:
                # A prior failed save may leave portable nodes or pending output.
                # Refresh and final validation above are the recovery receipt;
                # only now can a retry enter the ordinary save protocol.
                _lifecycle_event('recovered')
                _lifecycle_event('quiesce')
                _lifecycle_event('save_begin')
            grades.prepare_save()
            _lifecycle_event('save_ready')
        except Exception:
            _lifecycle_event('fail')
            scheduler.recover()
            grades.recover_flags()
            _recover_steps((grades.representations,))
            raise


    @persistent
    def save_finished_handler(*_args):
        # A fallback timer or duplicate callback may have already restored nodes.
        if not grades._saving:
            return
        output_guard = _application.component('graph.output_guard')
        # Guarded Save receives save_post before the native operator returns.
        # Its settle step (or the fallback timer) restores representations after
        # that reader boundary, including fresh verification of their new owners.
        if not output_guard.available():
            return
        try:
            grades.finish_save()
            _lifecycle_event('save_done')
        except Exception:
            _lifecycle_event('fail')
            scheduler = _application.component('graph.scheduler')
            scheduler.recover()
            grades.recover_flags()
            _recover_steps((grades.representations,))
            raise


    def clear_transient_caches():
        """Release cached RNA before replacement, without discovering new nodes."""
        locus = _application.component('ui.locus'); warp = _application.component('ui.warp'); chromatic_diagram = _application.component('ui.color')
        scopes = _application.component('scopes.sampling'); comparison = _application.component('scopes.reference')
        compositor = _application.component('graph.compositor')
        bake = _application.component('graph.bake')
        capture = _application.component('scopes.capture')
        capture_context = _application.component('scopes.context')
        _recover_steps((editing.clear, spaces.adapters.cache_clear, locus.clear,
                        warp.clear, _signatures.clear, cube_nodes.clear,
                        compositor.clear, bake.clear, capture.clear, capture_context.clear, scopes.clear, comparison.clear,
                        chromatic_diagram._DENSITY.clear), strict=True)


    def recovery_status():
        state=None if _recovery is None else _recovery.state
        phase='idle' if state is None else state.phase
        # Save/serialization can fail outside the reconciliation coordinator.
        # Expose the same explicit Retry without pretending work is in flight.
        if phase in ('idle','completed','superseded') and recovery_pending():phase='failed'
        return dict(state=phase,
                    generation=0 if state is None else state.generation,
                    completed=0 if state is None else state.settled,
                    restored=0 if state is None else state.restored,
                    total=0 if state is None else state.total,error=_recovery_error)


    def _begin_recovery(native):
        nonlocal _recovery,_recovery_work,_recovery_error
        from ..runtime.protocol import Protocol
        import time
        if _recovery is None:_recovery=Protocol(_application,'recovery')
        if _recovery.state.phase not in ('idle','superseded'):
            _recovery.event('supersede',generation=_recovery.state.generation)
        if _recovery_work is not None:
            for owner_ref in _recovery_work.get('owners',()):ownership.release(owner_ref)
        generation=_recovery.state.generation+1
        _recovery.require('begin','fence',generation=generation,native=native)
        _recovery.identity=dict(generation=generation)
        _recovery_error=''
        _recovery_work=dict(generation=generation,tokens=None,owners=(),deadline=time.monotonic()+5.)
        return _recovery_work


    def _fail_recovery(error):
        nonlocal _recovery_error
        from ...blender_runtime.execution.diagnostics import text as diagnostic_text
        _recovery_error=diagnostic_text(error)
        if _recovery is not None:_recovery.event('failed')
        ownership.fail()
        _lifecycle_event('fail')
        protocol=_application.app.lifecycle
        if protocol is not None and protocol.state.phase!='active':protocol.event('failed')
        # Do not release obligations or manufacture terminal worker receipts.
        # Tokens and the remaining worklist stay available to an explicit retry.
        LOG.error('Perceptual Grade runtime reconciliation failed: %s',_recovery_error)


    def _retirement_ready():
        runtime=_application.component('services').current_runtime()
        if runtime is not None and not runtime.closed:runtime.poll(budget_ms=4)
        checks=(cube_nodes.collect_retired(),
                _application.component('scopes.jobs').drain_retired(),
                _application.component('scopes.capture').collect_retired(),
                _application.component('graph.bake').physical_retired())
        return all(checks)


    def recovery_tick():
        """One bounded reconciliation step; all native output remains untouched."""
        if not _registered or _recovery is None or _recovery_work is None:return None
        import time
        work=_recovery_work;generation=work['generation']
        state=_recovery.state
        if state.generation!=generation or state.phase in ('completed','failed','superseded','replacing'):
            return None
        tokens=work['tokens']
        if tokens is None or not ownership.is_current(tokens):
            _recovery.event('supersede',generation=generation)
            return None
        try:
            guard=_application.component('graph.output_guard')
            guard.reconcile(cleanup=True)
            if state.phase=='retiring':
                if state.replaced and not work.get('guard_rebound'):
                    try:
                        guard.reset_after_native_replacement()
                    except guard.PublicationDeferred:
                        if time.monotonic()>=work['deadline']:
                            raise TimeoutError('Runtime recovery is waiting for native readers')
                        return .02
                    work['guard_rebound']=True
                ready=_retirement_ready()
                if not ready or not guard.available(cleanup=True):
                    if time.monotonic()>=work['deadline']:
                        raise TimeoutError('Runtime recovery is waiting for physical retirement or native readers')
                    return .02
                _recovery.require('retired','discover',workers=0,validated=True)
                return .001
            if not guard.available(cleanup=True):
                if time.monotonic()>=work['deadline']:
                    raise TimeoutError('Runtime recovery is waiting for native readers')
                return .02
            if state.phase=='discovering':
                # Only IDs and scalar controls are inspected. Retained owner
                # references are fenced by the same replacement generation.
                work['owners']=tuple(ownership.retain(node,'recovery.owner') for node in all_nodes()
                                     if grades.local(node.id_data) and not
                                     (work.get('initial') and cube_nodes.has_runtime_intent(node)))
                total=len(work['owners'])+1
                _recovery.require('discovered','adopt' if total>1 else 'reconcile',
                                  total=total,validated=True)
                return .001
            if state.phase=='adopting':
                node=ownership.resolve(work['owners'][state.settled])
                if node is None:raise ReferenceError('Recovery owner changed during reconciliation')
                cube_nodes.adopt_native(node,output_intent(node))
                _recovery.require('adopted','restore_wrapper' if state.settled+1==state.total-1 else 'adopt',
                                  index=state.settled,validated=True)
                return .001
            if state.phase=='restoring':
                previous = work['owners'][state.restored]
                node = ownership.resolve(previous)
                if node is None:raise ReferenceError('Saved control wrapper changed during restoration')
                restored = grades.restore_wrapper(node)
                if restored != node:
                    replacement = ownership.retain(restored, 'recovery.owner')
                    owners = list(work['owners'])
                    owners[state.restored] = replacement
                    work['owners'] = tuple(owners)
                    ownership.release(previous)
                _recovery.require('wrapper_restored',
                    'reconcile' if state.restored+1==state.total-1 else 'restore_wrapper',
                    index=state.restored,validated=True)
                return .001
            if state.phase=='reconciling':
                progress=_application.component('graph.bake').runtime_reconcile(max_items=1)
                work['bake_progress']=progress
                if progress.get('error'):raise RuntimeError(progress['error'])
                if not progress['done']:return .001
                _application.component('ownership.transactions').require_clean()
                if not _retirement_ready():
                    if time.monotonic()>=work['deadline']:raise TimeoutError('Recovery worker retirement exceeded five seconds')
                    return .02
                protocol=_application.app.lifecycle
                if protocol is not None:
                    receipt=dict(operation=protocol.state.operation,generation=protocol.state.generation)
                    if protocol.state.phase=='repairing':protocol.require('repaired','inspect',validated=True,debts=0,**receipt)
                    borrows=sum(len(arena._borrows) for arena in ownership.snapshots())
                    protocol.require('inspected','activate',validated=True,debts=0,readers=0,borrows=borrows,workers=0,**receipt)
                rebound=tuple((arena,token) for arena,token in tokens if arena.state.phase==2)
                if rebound:ownership.recovered(rebound)
                _lifecycle_event('recovered')
                _recovery.require('reconciled','activate',validated=True)
                for owner_ref in work['owners']:ownership.release(owner_ref)
                work['owners']=()
                _application.component('graph.scheduler').wake(poll_working_space,.001,persistent=True)
                grades.redraw()
                _application.component('integration.outcomes').recovered('perceptual_grade.retry_recovery')
                return None
            raise RuntimeError('Unexpected recovery phase: '+state.phase)
        except Exception as error:
            if isinstance(error,_application.component('graph.output_guard').PublicationDeferred):
                if time.monotonic()<work['deadline']:return .02
                error=TimeoutError('Runtime recovery is waiting for native readers')
            _fail_recovery(error)
            return None


    def retry_recovery():
        """Retry runtime inspection; never replay a saved compensation journal."""
        if not _registered:return False
        if _recovery is not None and _recovery.state.phase not in ('idle','completed','failed','superseded'):
            return False
        initial=_recovery is None and not recovery_pending()
        ownership.arena.default
        tokens=tuple((arena,arena.state.token) for arena in ownership.snapshots())
        work=_begin_recovery(False)
        work['tokens']=tokens
        work['initial']=initial
        try:
            if any(arena.state.phase==3 for arena,token in tokens):ownership.retry_recovery(tokens)
            protocol=_application.app.lifecycle
            if protocol is not None:
                if protocol.state.phase=='active':protocol.require('inspect','inspect')
                elif protocol.state.phase=='repair_needed':protocol.require('repair','repair')
            _lifecycle_event('recover')
            if not initial:_recover_steps((suspend_handler,clear_transient_caches),strict=True)
            _application.component('graph.scheduler').wake(recovery_tick,.001,persistent=True)
            return True
        except Exception as error:
            _fail_recovery(error)
            return False


    @persistent
    def undo_redo_pre_handler(*_args):
        """Fence runtime interests before native replacement, with no graph repair."""
        if not _registered:return
        nonlocal _replacement_pending
        _replacement_pending=True
        _begin_recovery(True)
        protocol=_application.app.lifecycle
        try:
            if protocol is not None:protocol.require('begin','fence',operation=protocol.state.operation+1)
            _application.component('graph.bake').begin_replacement()
            _lifecycle_event('recover')
            _recover_steps((suspend_handler,clear_transient_caches),strict=True)
        except Exception as error:
            _fail_recovery(error)
        finally:
            ownership.begin_boundary()


    @persistent
    def reset_cache_handler(*_args):
        """Native post acknowledges replacement; the bounded timer does adoption."""
        if not _registered:return
        nonlocal _replacement_pending
        if not _replacement_pending:
            retry_recovery()
            return
        tokens=ownership.native_replaced()
        if ownership.defer_pending_replacement():return
        _replacement_pending=False
        try:
            # A native replacement still happened when a pre-handler cleanup
            # failed. The retained pre/post boundary is its authority; retry
            # without that boundary never takes this path.
            if _recovery.state.phase=='failed':
                _begin_recovery(True)
                protocol=_application.app.lifecycle
                if protocol is not None:protocol.require('begin','fence',operation=protocol.state.operation+1)
            _lifecycle_event('recover')
            _recovery.require('replaced','retire_journals',native=True,validated=True)
            _application.component('ownership.transactions').native_replaced(tokens)
            _application.component('graph.output_resources').native_replaced()
            _application.component('graph.bake').retire_replaced_state()
            protocol=_application.app.lifecycle
            if protocol is not None:protocol.require('external_replaced','inspect',validated=True)
            _recovery_work['tokens']=tokens
            import time
            _recovery_work['deadline']=time.monotonic()+5.
            _application.component('graph.scheduler').wake(recovery_tick,.001,persistent=True)
        except Exception as error:
            _fail_recovery(error)


    def poll_working_space():
        if not _registered:
            return None
        _application.app.drain_finalizers()
        if not hasattr(bpy.data, 'node_groups'):
            return .1
        # Completions must be consumed even when recovery or the reader guard
        # would otherwise return before ordinary work is admitted.
        _application.component('graph.output_guard').reconcile()
        if _recovery is None:
            if not recovery_pending():retry_recovery()
            return .1
        if recovery_pending():
            return .1
        # Add-on enable runs with bpy.data replaced by _RestrictData. The first
        # scan belongs to a later application-timer tick, never to register().
        if not hasattr(bpy.data, 'node_groups'):
            return .1
        output_guard = _application.component('graph.output_guard')
        if not output_guard.available():
            return .05
        _application.component('graph.bake').sync_coordinates()
        jobs = _application.component('scopes.jobs')
        jobs.refresh_visible()
        refresh_all()
        diagram = _application.component('ui.diagram')
        diagram.ensure_routers()
        return None if bpy.app.background else .35


    def assign_kernel(node,kernel):
        """Select a numerical branch variant without losing instance links/values."""
        transactions = _application.component('ownership.transactions')
        tree=node.id_data
        if tree.is_evaluated or not grades.local(tree):
            raise ValueError('Cannot replace a borrowed or linked node kernel')
        record=transactions.kernel_snapshot(node)
        record['after_kernel']=transactions.id_key(kernel)
        tx=transactions.Transaction('kernel',record)
        def install():
            for link in tuple(tree.links):
                if link.from_node==node or link.to_node==node:tree.links.remove(link)
            node.node_tree=kernel
            if not _application.component('graph.output_resources').unique_owner(node):
                raise ValueError('Kernel assignment requires private output resources')
            _application.component('graph.bake').materialize_binding(node,kernel)
            for name,value in record['values'].items():
                if name in node.inputs:node.inputs[name].default_value=value
            for source,output,target,input_ in record['links']:
                src,dst=transactions.find_node(tree,source),transactions.find_node(tree,target)
                if src is None or dst is None:raise RuntimeError('Kernel assignment lost a linked node')
                tree.links.new(src.outputs[output],dst.inputs[input_])
        try:
            tx.apply('kernel_and_links_replace',install,lambda:transactions.repair_kernel(record),
                     before_edges=(record['kernel'],),after_edges=(transactions.id_key(kernel),))
            tx.commit()
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()


    def make_kernel(tree_type):
        return cube_nodes.make_kernel(tree_type,kind='FORWARD')


    def initialize_grade(node):
        space = working_space()
        node.node_tree = make_kernel(node.id_data.bl_idname)
        node.name = tr('node.grade.title')
        node.label = tr('node.grade.title')
        node['_pg_auto_label'] = node.label
        node.width = 220
        node.show_options = True
        node.use_custom_color = True;node.color = (.13, .28, .29)
        node[TAG] = VERSION;node['_pg_space'] = space
        node[grades.NODE_SCHEMA] = grades.SCHEMA
        grades.assign_values(node, grades.defaults())
        for key in ('_pg_pending_init', '_pg_pending_copy'):
            if key in node:del node[key]
        refresh_node(node, force=True)


    def create_grade(tree):
        if tree.bl_idname not in grades.KINDS or tree.get(KERNEL):
            raise ValueError(tr('error.add_location'))
        node = tree.nodes.new(grades.KINDS[tree.bl_idname])
        initialize_grade(node)
        for other in tree.nodes:
            other.select = other == node
        tree.nodes.active = node
        return node


    def register():
        nonlocal _registered, _class_registered
        _application.register_class(PGLinearSettings)
        _class_registered = True
        try:
            _application.register_class(PGToneSettings)
            _application.register_class(PGChromaticSettings)
            bpy.types.Node.pg_linear = PointerProperty(type=PGLinearSettings)
            bpy.types.Node.pg_tone = PointerProperty(type=PGToneSettings)
            bpy.types.Node.pg_chromatic = PointerProperty(type=PGChromaticSettings)
            grades.register()
            stacks = _application.component('graph.stacks')
            stacks.register()
            bake = _application.component('graph.bake')
            bake.register()
            compositor = _application.component('graph.compositor')
            # Force the actor/store construction on this ordinary foreground
            # path, before any callback or passive draw can access the guard.
            _application.component('graph.output_guard').observation()
            _application.component('scopes.automatic').register()
            for name, fn in (('depsgraph_update_post', sync_handler), ('frame_change_post', sync_handler),
                             ('composite_pre', compositor.composite_pre), ('composite_post', compositor.composite_post), ('composite_cancel', compositor.composite_cancel),
                             ('render_init', render_init_handler), ('render_complete', render_complete_handler), ('render_cancel', render_complete_handler),
                             ('render_pre', render_handler), ('render_post', render_end_handler), ('render_post', scope_render_handler),
                             ('save_pre', save_handler), ('save_post', save_finished_handler), ('save_post_fail', save_finished_handler),
                             ('load_pre', undo_redo_pre_handler), ('undo_pre', undo_redo_pre_handler), ('redo_pre', undo_redo_pre_handler), ('load_post', reset_cache_handler),
                             ('undo_post', reset_cache_handler), ('redo_post', reset_cache_handler)):
                _application.add_handler(name, fn)
            _signatures.clear()
            _registered = True
            _lifecycle_event('activate')
            # Also schedule the initial scan in background mode. Render/load/frame
            # handlers still synchronize grades if no timer tick occurs before use.
            scheduler = _application.component('graph.scheduler')
            scheduler.wake(poll_working_space,.1,persistent=True)
        except Exception:
            unregister()
            raise


    def unregister():
        nonlocal _registered, _class_registered
        grades.prepare_disable()
        _registered = False
        compositor = _application.component('graph.compositor'); scheduler = _application.component('graph.scheduler')
        scheduler.unwake(poll_working_space)
        scheduler.unwake(recovery_tick)
        for name, fn in (('depsgraph_update_post', sync_handler), ('frame_change_post', sync_handler),
                         ('composite_pre', compositor.composite_pre), ('composite_post', compositor.composite_post), ('composite_cancel', compositor.composite_cancel),
                         ('render_init', render_init_handler), ('render_complete', render_complete_handler), ('render_cancel', render_complete_handler),
                         ('render_pre', render_handler), ('render_post', render_end_handler), ('render_post', scope_render_handler),
                         ('save_pre', save_handler), ('save_post', save_finished_handler), ('save_post_fail', save_finished_handler),
                         ('load_pre', undo_redo_pre_handler), ('undo_pre', undo_redo_pre_handler), ('redo_pre', undo_redo_pre_handler), ('load_post', reset_cache_handler),
                         ('undo_post', reset_cache_handler), ('redo_post', reset_cache_handler)):
            _application.remove_handler(name, fn)
        # Preserve the native group graph and its last row uniforms. Saved grades
        # continue evaluating with the add-on disabled; animation then needs it on.
        grades.unregister()
        stacks = _application.component('graph.stacks')
        stacks.unregister()
        bake = _application.component('graph.bake')
        bake.unregister()
        if hasattr(bpy.types.Node, 'pg_linear'):
            del bpy.types.Node.pg_linear
        if hasattr(bpy.types.Node,'pg_chromatic'):del bpy.types.Node.pg_chromatic
        if bpy.types.PropertyGroup.bl_rna_get_subclass_py('PGChromaticSettings',None) is PGChromaticSettings:
            _application.unregister_class(PGChromaticSettings)
        if hasattr(bpy.types.Node,'pg_tone'):del bpy.types.Node.pg_tone
        if bpy.types.PropertyGroup.bl_rna_get_subclass_py('PGToneSettings',None) is PGToneSettings:
            _application.unregister_class(PGToneSettings)
        if _class_registered:
            _application.unregister_class(PGLinearSettings)
            _class_registered = False
        _signatures.clear();_updating.clear();cube_nodes.clear()
        spaces.adapters.cache_clear()
        compositor = _application.component('graph.compositor')
        compositor.clear()
        scheduler.shutdown()

    def _set_export_KERNEL(value):
        nonlocal KERNEL
        KERNEL = value

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export_RELEASE(value):
        nonlocal RELEASE
        RELEASE = value

    def _set_export_TAG(value):
        nonlocal TAG
        TAG = value

    def _set_export_VERSION(value):
        nonlocal VERSION
        VERSION = value

    def _set_export__class_registered(value):
        nonlocal _class_registered
        _class_registered = value

    def _set_export__registered(value):
        nonlocal _registered
        _registered = value

    def _set_export__rendering(value):
        nonlocal _rendering
        _rendering = value

    def _set_export__signatures(value):
        nonlocal _signatures
        _signatures = value

    def _set_export__updating(value):
        nonlocal _updating
        _updating = value

    def _set_export_args(value):
        nonlocal args
        args = value

    def _set_export_factory(value):
        nonlocal factory
        factory = value

    def _set_export__lifecycle_event(value):
        nonlocal _lifecycle_event
        _lifecycle_event = value

    def _set_export_recovery_pending(value):
        nonlocal recovery_pending
        recovery_pending = value

    def _set_export_working_space(value):
        nonlocal working_space
        working_space = value

    def _set_export_is_grade(value):
        nonlocal is_grade
        is_grade = value

    def _set_export_resource_action(value):
        nonlocal resource_action
        resource_action = value

    def _set_export_resources_ready(value):
        nonlocal resources_ready
        resources_ready = value

    def _set_export_copy_output_resources(value):
        nonlocal copy_output_resources
        copy_output_resources = value

    def _set_export_prepare_output_resources(value):
        nonlocal prepare_output_resources
        prepare_output_resources = value

    def _set_export_active_grade(value):
        nonlocal active_grade
        active_grade = value

    def _set_export_editable(value):
        nonlocal editable
        editable = value

    def _set_export_read_parameters(value):
        nonlocal read_parameters
        read_parameters = value

    def _set_export_grading_space(value):
        nonlocal grading_space
        grading_space = value

    def _set_export_linear_matrix(value):
        nonlocal linear_matrix
        linear_matrix = value

    def _set_export_read_tone(value):
        nonlocal read_tone
        read_tone = value

    def _set_export_read_chromatic(value):
        nonlocal read_chromatic
        read_chromatic = value

    def _set_export_color_model(value):
        nonlocal color_model
        color_model = value

    def _set_export_owner(value):
        nonlocal owner
        owner = value

    def _set_export_changed(value):
        nonlocal changed
        changed = value

    def _set_export_settings(value):
        nonlocal settings
        settings = value

    def _set_export_bake_model(value):
        nonlocal bake_model
        bake_model = value

    def _set_export_PGLinearSettings(value):
        nonlocal PGLinearSettings
        PGLinearSettings = value

    def _set_export_linear_property(value):
        nonlocal linear_property
        linear_property = value

    def _set_export_tone_callback(value):
        nonlocal tone_callback
        tone_callback = value

    def _set_export_reference_get(value):
        nonlocal reference_get
        reference_get = value

    def _set_export_reference_set(value):
        nonlocal reference_set
        reference_set = value

    def _set_export_PGToneSettings(value):
        nonlocal PGToneSettings
        PGToneSettings = value

    def _set_export_focus_changed(value):
        nonlocal focus_changed
        focus_changed = value

    def _set_export_PGChromaticSettings(value):
        nonlocal PGChromaticSettings
        PGChromaticSettings = value

    def _set_export_write_chromatic(value):
        nonlocal write_chromatic
        write_chromatic = value

    def _set_export_keyframe_chromatic(value):
        nonlocal keyframe_chromatic
        keyframe_chromatic = value

    def _set_export_write_tone(value):
        nonlocal write_tone
        write_tone = value

    def _set_export_keyframe_tone(value):
        nonlocal keyframe_tone
        keyframe_tone = value

    def _set_export_write_values(value):
        nonlocal write_values
        write_values = value

    def _set_export_keyframe_parameter(value):
        nonlocal keyframe_parameter
        keyframe_parameter = value

    def _set_export_reset_linear(value):
        nonlocal reset_linear
        reset_linear = value

    def _set_export_refresh_node(value):
        nonlocal refresh_node
        refresh_node = value

    def _set_export_set_uniform(value):
        nonlocal set_uniform
        set_uniform = value

    def _set_export_all_grades(value):
        nonlocal all_grades
        all_grades = value

    def _set_export_all_nodes(value):
        nonlocal all_nodes
        all_nodes = value

    def _set_export_refresh_all(value):
        nonlocal refresh_all
        refresh_all = value

    def _set_export_sync_handler(value):
        nonlocal sync_handler
        sync_handler = value

    def _set_export_render_handler(value):
        nonlocal render_handler
        render_handler = value

    def _set_export_render_end_handler(value):
        nonlocal render_end_handler
        render_end_handler = value

    def _set_export_scope_render_handler(value):
        nonlocal scope_render_handler
        scope_render_handler = value

    def _set_export__recover_steps(value):
        nonlocal _recover_steps
        _recover_steps = value

    def _set_export_suspend_handler(value):
        nonlocal suspend_handler
        suspend_handler = value

    def _set_export_save_handler(value):
        nonlocal save_handler
        save_handler = value

    def _set_export_save_finished_handler(value):
        nonlocal save_finished_handler
        save_finished_handler = value

    def _set_export_clear_transient_caches(value):
        nonlocal clear_transient_caches
        clear_transient_caches = value

    def _set_export_undo_redo_pre_handler(value):
        nonlocal undo_redo_pre_handler
        undo_redo_pre_handler = value

    def _set_export_reset_cache_handler(value):
        nonlocal reset_cache_handler
        reset_cache_handler = value

    def _set_export_poll_working_space(value):
        nonlocal poll_working_space
        poll_working_space = value

    def _set_export_assign_kernel(value):
        nonlocal assign_kernel
        assign_kernel = value

    def _set_export_make_kernel(value):
        nonlocal make_kernel
        make_kernel = value

    def _set_export_initialize_grade(value):
        nonlocal initialize_grade
        initialize_grade = value

    def _set_export_create_grade(value):
        nonlocal create_grade
        create_grade = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CONTEXT': lambda: CONTEXT,
        'EnumProperty': lambda: EnumProperty,
        'FloatProperty': lambda: FloatProperty,
        'IntProperty': lambda: IntProperty,
        'KERNEL': lambda: KERNEL,
        'LOG': lambda: LOG,
        'PGChromaticSettings': lambda: PGChromaticSettings,
        'PGLinearSettings': lambda: PGLinearSettings,
        'PGToneSettings': lambda: PGToneSettings,
        'PointerProperty': lambda: PointerProperty,
        'RELEASE': lambda: RELEASE,
        'TAG': lambda: TAG,
        'VERSION': lambda: VERSION,
        '_class_registered': lambda: _class_registered,
        '_lifecycle_event': lambda: _lifecycle_event,
        '_name': lambda: _name,
        '_recover_steps': lambda: _recover_steps,
        '_registered': lambda: _registered,
        '_rendering': lambda: _rendering,
        '_signatures': lambda: _signatures,
        '_updating': lambda: _updating,
        'active_grade': lambda: active_grade,
        'all_grades': lambda: all_grades,
        'all_nodes': lambda: all_nodes,
        'args': lambda: args,
        'assign_kernel': lambda: assign_kernel,
        'bake_model': lambda: bake_model,
        'bpy': lambda: bpy,
        'ch': lambda: ch,
        'changed': lambda: changed,
        'clear_transient_caches': lambda: clear_transient_caches,
        'cm': lambda: cm,
        'color_model': lambda: color_model,
        'control_tooltip': lambda: control_tooltip,
        'controls': lambda: controls,
        'copy_output_resources': lambda: copy_output_resources,
        'create_grade': lambda: create_grade,
        'cube_nodes': lambda: cube_nodes,
        'dctl': lambda: dctl,
        'default': lambda: default,
        'editable': lambda: editable,
        'editing': lambda: editing,
        'factory': lambda: factory,
        'focus_changed': lambda: focus_changed,
        'grades': lambda: grades,
        'grading_space': lambda: grading_space,
        'hi': lambda: hi,
        'initialize_grade': lambda: initialize_grade,
        'is_grade': lambda: is_grade,
        'keyframe_chromatic': lambda: keyframe_chromatic,
        'keyframe_parameter': lambda: keyframe_parameter,
        'keyframe_tone': lambda: keyframe_tone,
        'label': lambda: label,
        'linear_matrix': lambda: linear_matrix,
        'linear_property': lambda: linear_property,
        'lo': lambda: lo,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'make_kernel': lambda: make_kernel,
        'name': lambda: name,
        'native': lambda: native,
        'owner': lambda: owner,
        'ownership': lambda: ownership,
        'persistent': lambda: persistent,
        'poll_working_space': lambda: poll_working_space,
        'precision': lambda: precision,
        'prepare_output_resources': lambda: prepare_output_resources,
        'read_chromatic': lambda: read_chromatic,
        'read_parameters': lambda: read_parameters,
        'read_tone': lambda: read_tone,
        'recovery_pending': lambda: recovery_pending,
        'recovery_status': lambda: recovery_status,
        'recovery_tick': lambda: recovery_tick,
        'retry_recovery': lambda: retry_recovery,
        'output_intent': lambda: output_intent,
        'observe_intent': lambda: observe_intent,
        'reference_get': lambda: reference_get,
        'reference_set': lambda: reference_set,
        'refresh_all': lambda: refresh_all,
        'refresh_node': lambda: refresh_node,
        'register': lambda: register,
        'render_end_handler': lambda: render_end_handler,
        'render_handler': lambda: render_handler,
        'render_init_handler': lambda: render_init_handler,
        'render_complete_handler': lambda: render_complete_handler,
        'reset_cache_handler': lambda: reset_cache_handler,
        'reset_linear': lambda: reset_linear,
        'resource_action': lambda: resource_action,
        'resources_ready': lambda: resources_ready,
        'save_finished_handler': lambda: save_finished_handler,
        'save_handler': lambda: save_handler,
        'scope_render_handler': lambda: scope_render_handler,
        'set_uniform': lambda: set_uniform,
        'settings': lambda: settings,
        'spaces': lambda: spaces,
        'suspend_handler': lambda: suspend_handler,
        'sync_handler': lambda: sync_handler,
        'tm': lambda: tm,
        'tone_callback': lambda: tone_callback,
        'tr': lambda: tr,
        'transforms': lambda: transforms,
        'undo_redo_pre_handler': lambda: undo_redo_pre_handler,
        'unregister': lambda: unregister,
        'working_space': lambda: working_space,
        'write_chromatic': lambda: write_chromatic,
        'write_tone': lambda: write_tone,
        'write_values': lambda: write_values,
    }, writers={
        'KERNEL': _set_export_KERNEL,
        'LOG': _set_export_LOG,
        'RELEASE': _set_export_RELEASE,
        'TAG': _set_export_TAG,
        'VERSION': _set_export_VERSION,
        '_class_registered': _set_export__class_registered,
        '_registered': _set_export__registered,
        '_rendering': _set_export__rendering,
        '_signatures': _set_export__signatures,
        '_updating': _set_export__updating,
        'args': _set_export_args,
        'factory': _set_export_factory,
    '_lifecycle_event': _set_export__lifecycle_event, 'recovery_pending': _set_export_recovery_pending, 'working_space': _set_export_working_space, 'is_grade': _set_export_is_grade, 'resource_action': _set_export_resource_action, 'resources_ready': _set_export_resources_ready, 'copy_output_resources': _set_export_copy_output_resources, 'prepare_output_resources': _set_export_prepare_output_resources, 'active_grade': _set_export_active_grade, 'editable': _set_export_editable, 'read_parameters': _set_export_read_parameters, 'grading_space': _set_export_grading_space, 'linear_matrix': _set_export_linear_matrix, 'read_tone': _set_export_read_tone, 'read_chromatic': _set_export_read_chromatic, 'color_model': _set_export_color_model, 'owner': _set_export_owner, 'changed': _set_export_changed, 'settings': _set_export_settings, 'bake_model': _set_export_bake_model, 'PGLinearSettings': _set_export_PGLinearSettings, 'linear_property': _set_export_linear_property, 'tone_callback': _set_export_tone_callback, 'reference_get': _set_export_reference_get, 'reference_set': _set_export_reference_set, 'PGToneSettings': _set_export_PGToneSettings, 'focus_changed': _set_export_focus_changed, 'PGChromaticSettings': _set_export_PGChromaticSettings, 'write_chromatic': _set_export_write_chromatic, 'keyframe_chromatic': _set_export_keyframe_chromatic, 'write_tone': _set_export_write_tone, 'keyframe_tone': _set_export_keyframe_tone, 'write_values': _set_export_write_values, 'keyframe_parameter': _set_export_keyframe_parameter, 'reset_linear': _set_export_reset_linear, 'refresh_node': _set_export_refresh_node, 'set_uniform': _set_export_set_uniform, 'all_grades': _set_export_all_grades, 'all_nodes': _set_export_all_nodes, 'refresh_all': _set_export_refresh_all, 'sync_handler': _set_export_sync_handler, 'render_handler': _set_export_render_handler, 'render_end_handler': _set_export_render_end_handler, 'scope_render_handler': _set_export_scope_render_handler, '_recover_steps': _set_export__recover_steps, 'suspend_handler': _set_export_suspend_handler, 'save_handler': _set_export_save_handler, 'save_finished_handler': _set_export_save_finished_handler, 'clear_transient_caches': _set_export_clear_transient_caches, 'undo_redo_pre_handler': _set_export_undo_redo_pre_handler, 'reset_cache_handler': _set_export_reset_cache_handler, 'poll_working_space': _set_export_poll_working_space, 'assign_kernel': _set_export_assign_kernel, 'make_kernel': _set_export_make_kernel, 'initialize_grade': _set_export_initialize_grade, 'create_grade': _set_export_create_grade, 'register': _set_export_register, 'unregister': _set_export_unregister})
