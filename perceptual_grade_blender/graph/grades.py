"""Shared Grade datablocks and their per-node evaluation/UI adapters.

Only model parameters belong to a Grade. Node PropertyGroups hold local edit
state for empty selectors and handle gestures; a selected Grade owns the values.
"""

from ..integration.components import Component

def create(_application):
    from contextlib import contextmanager
    import json
    import logging

    import bpy
    from bpy.props import PointerProperty, EnumProperty, StringProperty

    native = _application.component('localization.i18n').native; tr = _application.component('localization.i18n').tr; error_text = _application.component('localization.i18n').error_text; CONTEXT = _application.component('localization.i18n').CONTEXT
    from ...perceptual_grade_core.model import linear, tone, color
    from ...color_core.linear import WHITES, rebase
    from ...blender_runtime.execution.diagnostics import log_exception

    LOG = logging.getLogger(__package__)

    TAG = '_pg_grade_data'
    SCHEMA = 8
    NODE_SCHEMA = '_pg_shared_schema'
    KINDS = {'ShaderNodeTree': 'PGShaderGrade', 'CompositorNodeTree': 'PGCompositorGrade'}
    FAMILIES = (('pg_tone', tone.PARAM_NAMES), ('pg_chromatic', color.PARAM_NAMES))
    FOCUS_FIELDS=('family','active_tone','active_chroma','active_handle')
    MATRIX = '_pg_linear_matrix'
    MATRIX_WHITE = '_pg_matrix_white'
    WHITE = '_pg_working_white'
    WRAPPER_STATE = '_pg_portable_wrapper'
    _busy = set()
    _converting = False
    _saving = False
    _save_wrappers = {}
    _classes = []


    def core():
        nodes = _application.component('graph.nodes')
        return nodes


    def is_data(value):
        return isinstance(value, bpy.types.NodeTree) and value.get(TAG) == SCHEMA


    def selected(node):
        value = getattr(node, 'pg_grade', None)
        return value if is_data(value) else None


    def data_owner(value):
        return value if is_data(value) else selected(value) or value


    def local(value):
        return value.library is None or value.override_library is not None


    def editable(node):
        editing = _application.component('graph.editing')
        if editing.is_target(node):return editing.valid(node)
        return local(node.id_data) and (selected(node) is None or local(selected(node)))


    @contextmanager
    def guarded(value):
        pointer = value.as_pointer()
        already = pointer in _busy
        _busy.add(pointer)
        try:
            yield
        finally:
            if not already:
                _busy.discard(pointer)


    def release_node_references(node):
        """Release owned Grade pointers before deleting a node, exactly once.

        Blender 5.2 does not decrement IDs held by extension RNA properties when
        removing an individual node. Clear them through RNA before removal; never
        adjust ID user counts directly. Evaluated copies do not own these users.
        """
        if node.id_data.is_evaluated:
            return
        scheduler = _application.component('graph.scheduler')
        retire = _application.component('ownership').retire
        try:
            try:
                scheduler.invalidate(node)
            finally:
                with guarded(node):
                    try:
                        if getattr(node, 'pg_grade', None) is not None:
                            node.pg_grade = None
                    finally:
                        entries = getattr(node, 'pg_stack_entries', None)
                        if entries is not None and len(entries):
                            entries.clear()
        finally:
            try:
                if getattr(node, 'pg_bake_image', None) is not None:
                    bake = _application.component('graph.bake')
                    with bake.binding_suppressed(node):
                        node.pg_bake_image = None
            finally:
                retire(node)


    def snapshot(value):
        result = {family: {name: getattr(getattr(value, family), name) for name in names}
                  for family, names in FAMILIES}
        result.update(matrix=tuple(value.get(MATRIX, sum(linear.I3, ()))),
                      matrix_white=value.get(MATRIX_WHITE, 'D65'), white=white(value))
        return result


    def assign_values(value, values):
        with guarded(value):
            for family, fields in values.items():
                if family not in dict(FAMILIES):
                    continue
                target = getattr(value, family)
                for name, setting in fields.items():
                    if getattr(target, name) != setting:
                        setattr(target, name, setting)
            if tuple(value.get(MATRIX, ())) != values['matrix']:
                value[MATRIX] = list(values['matrix'])
            if value.get(MATRIX_WHITE) != values['matrix_white']:
                value[MATRIX_WHITE] = values['matrix_white']
            if value.get(WHITE) != values['white']:
                value[WHITE] = values['white']


    def defaults():
        t, c = tone.Parameters(), color.Parameters()
        result = {family: {name: str(getattr(model, name)) if name == 'peak_nits'
                          else getattr(model, name) for name in names}
                  for (family, names), model in zip(FAMILIES, (t, c))}
        result.update(matrix=sum(linear.I3, ()), matrix_white='D65', white='D65')
        return result


    def white(value):
        return data_owner(value).get(WHITE, 'D65')


    def space(value):
        return linear.GradeSpace(white(value))


    def matrix(value):
        owner = data_owner(value)
        flat = owner.get(MATRIX, sum(linear.I3, ()))
        result = tuple(tuple(float(flat[3*i+j]) for j in range(3)) for i in range(3))
        return rebase(result, WHITES[owner.get(MATRIX_WHITE, 'D65')],
                             WHITES[white(owner)])


    def write_matrix(value, matrix):
        """Write current-white XYZ into the matrix's recorded animation basis."""
        owner = data_owner(value)
        result = rebase(matrix, WHITES[white(owner)],
                               WHITES[owner.get(MATRIX_WHITE, 'D65')])
        owner[MATRIX] = [v for row in result for v in row]


    def parameters(value):
        return linear.decompose(matrix(value), space(value))


    def edit_linear(value, values):
        if not (local(value) if is_data(value) else editable(value)):
            return
        current = matrix(value)
        result = linear.edited_matrix(current, space(value), values)
        if result == current:
            return
        target = value if is_data(value) else ensure(value)
        write_matrix(target, result)
        publish(target)


    def white_get(value):
        return tuple(WHITES).index(white(value))


    def white_set(value, index):
        if not (local(value) if is_data(value) else editable(value)):
            return
        preset = tuple(WHITES)[index]
        if preset != 'D65':raise ValueError('Only cached D65 / 203 nits is supported')
        if preset == white(value):
            return
        target = value if is_data(value) else ensure(value)
        transformed = rebase(matrix(target), WHITES[white(target)], WHITES[preset])
        target[WHITE] = preset
        # Animated coefficients retain their recorded basis: conjugation then
        # applies to every interpolated frame, Action, NLA strip and driver.
        animated = any(c.data_path.startswith('["'+MATRIX+'"]') for c in owner_curves(target))
        if not animated:
            target[MATRIX_WHITE] = preset
            target[MATRIX] = [x for row in transformed for x in row]
        publish(target)


    def users(grade):
        return [node for node in core().all_grades() if selected(node) == grade]


    def action_curves(action):
        """Iterate the curves in a layered Action."""
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', ()):
                    yield from bag.fcurves


    def owner_curves(owner):
        """Channels that node removal would otherwise delete, including NLA."""
        animation = owner.animation_data
        if animation is None:
            return
        yield from animation.drivers
        actions = set()
        if animation.action is not None:
            actions.add(animation.action)
        def collect(strips):
            for strip in strips:
                if strip.action is not None:
                    actions.add(strip.action)
                collect(getattr(strip, 'strips', ()))
        for track in animation.nla_tracks:
            collect(track.strips)
        for action in actions:
            yield from action_curves(action)


    def sync_node(node):
        grade = selected(node)
        if grade is not None:
            assign_values(node, snapshot(grade))


    def bind_animation_dependency(node):
        """Let Blender evaluate the Grade's Action/NLA/drivers in the scene graph.

        A Python-only ID reference is insufficient for a custom NodeTree's
        animation. This unused local value declares a real dependency; Blender,
        rather than a Python F-curve sampler, remains the animation evaluator.
        """
        grade = selected(node)
        path = '["_pg_grade_dependency"]'
        if grade is None:
            if '_pg_grade_dependency' in node:
                node.driver_remove(path)
            return
        node['_pg_grade_dependency'] = 0.
        curve = node.driver_add(path)
        driver = curve.driver
        driver.type = 'SCRIPTED'
        for variable in list(driver.variables):
            driver.variables.remove(variable)
        variable = driver.variables.new()
        variable.name = 'grade_value';variable.type = 'SINGLE_PROP'
        target = variable.targets[0]
        target.id_type = 'NODETREE';target.id = grade
        target.data_path = '["'+MATRIX+'"][0]'
        driver.expression = '0.0 * grade_value'


    def redraw():
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()


    def publish(grade, force=False):
        if core().recovery_pending():return
        for node in users(grade):
            if local(node.id_data):
                sync_node(node)
                core().refresh_node(node, force=force)
        stacks = _application.component('graph.stacks')
        stacks.publish(grade)
        redraw()


    def new(node, duplicate=True):
        """Explicit New: copy the selected Grade, otherwise capture local defaults."""
        transactions = _application.component('ownership.transactions')
        if node.id_data.is_evaluated or not local(node.id_data):
            raise ValueError(tr('warning.local_tree'))
        transactions.require_clean()
        old = selected(node)
        copying = old is not None and duplicate
        old_action = old.animation_data.action if copying and old.animation_data else None
        path = node.path_from_id() + '["_pg_grade_dependency"]'
        animation = node.id_data.animation_data
        curve = animation.drivers.find(path) if animation else None
        record = {'node': transactions.node_key(node), 'before_grade': transactions.id_key(old),
                  'before_values': snapshot(node), 'after_values': snapshot(old if copying else node),
                  'driver_present': curve is not None, 'driver_mute': bool(curve.mute) if curve else False,
                  'dependency_present': '_pg_grade_dependency' in node,
                  'dependency_value': node.get('_pg_grade_dependency', 0.),
                  'old_action': transactions.id_key(old_action),
                  'created_grade': None, 'created_action': None,
                  'grade_resolved': False, 'action_resolved': old_action is None,
                  'existing_grades': tuple(transactions.id_key(value) for value in bpy.data.node_groups),
                  'existing_actions': tuple(transactions.id_key(value) for value in bpy.data.actions)}
        tx = transactions.Transaction('grade_create', record)
        try:
            grade = tx.apply('grade_allocate',
                             lambda: old.copy() if copying else bpy.data.node_groups.new('Grade.001', PGGradeData.bl_idname),
                             lambda: _discard_created_grade(tx.record), retirement=True,
                             after_edges=(transactions.id_key(old_action),) if old_action else ())
            tx.record['created_grade'] = transactions.fresh_identity(grade)
            tx.record['grade_resolved'] = True
            if not copying:
                def initialize():
                    grade[TAG] = SCHEMA
                    assign_values(grade, tx.record['after_values'])
                tx.apply('grade_initialize', initialize, lambda: None)
            if old_action is not None:
                action = tx.apply('grade_action_allocate', lambda: old_action.copy(),
                                  lambda: _discard_created_action(tx.record), retirement=True)
                tx.record['created_action'] = transactions.fresh_identity(action)
                tx.record['action_resolved'] = True
                tx.apply('grade_action_assign', lambda: setattr(grade.animation_data, 'action', action),
                         lambda: _restore_created_action_link(tx.record),
                         before_edges=(transactions.id_key(old_action),),
                         after_edges=(transactions.id_key(action),))
            had_fake_user = bool(grade.use_fake_user)
            tx.apply('grade_fake_user', lambda: setattr(grade, 'use_fake_user', True),
                     lambda: setattr(grade, 'use_fake_user', had_fake_user),
                     before_edges=(transactions.id_key(grade),) if had_fake_user else (),
                     after_edges=(transactions.id_key(grade),))
            def select_and_bind():
                with guarded(node): node.pg_grade = grade
                bind_animation_dependency(node)
            tx.apply('grade_selector_and_dependency', select_and_bind,
                     lambda: _restore_grade_selector(tx.record),
                     before_edges=(transactions.id_key(old),) if old else (),
                     after_edges=(transactions.id_key(grade),))
            tx.apply('grade_node_values', lambda: sync_node(node),
                     lambda: assign_values(node, tx.record['before_values']))
            verify_grade_creation(tx.record, after=True)
            tx.commit()
            return grade
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()


    def _created_datablock(record, action=False):
        transactions = _application.component('ownership.transactions')
        field, resolved, inventory = ('created_action', 'action_resolved', 'existing_actions') if action else (
            'created_grade', 'grade_resolved', 'existing_grades')
        find = transactions.find_action if action else transactions.find_tree
        created = find(record[field])
        if created is not None or record[resolved]: return created
        known = set()
        for identity in record[inventory]:
            value = find(identity)
            if value is not None: known.add(value.session_uid)
        values = bpy.data.actions if action else bpy.data.node_groups
        candidates = [value for value in values if value.session_uid not in known]
        if len(candidates) > 1 or (candidates and not action and candidates[0].bl_idname != PGGradeData.bl_idname):
            raise RuntimeError('Native Grade allocation has ambiguous partial effects')
        if candidates:
            created = candidates[0]
            record[field] = transactions.fresh_identity(created)
        record[resolved] = True
        return created


    def _creation_node(record):
        transactions = _application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        if tree is None: return None
        node = transactions.find_node(tree, record['node'])
        if node is None: raise RuntimeError('Grade creation repair lost its selector node')
        return node


    def _restore_grade_selector(record):
        transactions = _application.component('ownership.transactions')
        node = _creation_node(record)
        if node is None: return
        before = None if record['before_grade'] is None else transactions.require_tree(record['before_grade'])
        with guarded(node): node.pg_grade = before
        path = '["_pg_grade_dependency"]'
        if record['driver_present']:
            bind_animation_dependency(node)
            curve = node.id_data.animation_data.drivers.find(node.path_from_id() + path)
            if curve is None: raise RuntimeError('Grade repair could not restore its animation dependency')
            curve.mute = record['driver_mute']
        else:
            animation = node.id_data.animation_data
            if animation and animation.drivers.find(node.path_from_id() + path): node.driver_remove(path)
        if record['dependency_present']:
            node['_pg_grade_dependency'] = record['dependency_value']
        elif '_pg_grade_dependency' in node:
            del node['_pg_grade_dependency']


    def _restore_created_action_link(record):
        transactions = _application.component('ownership.transactions')
        grade = _created_datablock(record)
        if grade is None: return
        action = transactions.find_action(record['old_action'])
        if record['old_action'] is not None and action is None:
            raise RuntimeError('Grade repair lost the original Action')
        if grade.animation_data is not None: grade.animation_data.action = action


    def _discard_created_grade(record):
        native = _application.app.blender.native
        grade = _created_datablock(record)
        if grade is None: return
        grade.use_fake_user = False
        if grade.users:
            raise RuntimeError('A created Grade still has native owners')
        native.remove_node_group(grade)


    def _discard_created_action(record):
        native = _application.app.blender.native
        action = _created_datablock(record, action=True)
        if action is None: return
        action.use_fake_user = False
        if action.users:
            raise RuntimeError('A created Action still has native owners')
        native.remove_action(action)


    def repair_grade_creation(record):
        node = _creation_node(record)
        if node is not None:
            failed = False
            try: _restore_grade_selector(record)
            except Exception: failed = True
            try: assign_values(node, record['before_values'])
            except Exception: failed = True
            if failed: raise RuntimeError('Grade selector repair remains incomplete')
        _discard_created_grade(record)
        _discard_created_action(record)
        verify_grade_creation(record)


    def verify_grade_creation(record, after=False):
        transactions = _application.component('ownership.transactions')
        node = _creation_node(record)
        if node is None: return
        identity = record['created_grade'] if after else record['before_grade']
        expected = None if identity is None else transactions.require_tree(identity)
        if selected(node) != expected or snapshot(node) != record['after_values' if after else 'before_values']:
            raise RuntimeError('Grade selector or values differ from the ownership receipt')
        path = node.path_from_id() + '["_pg_grade_dependency"]'
        animation = node.id_data.animation_data
        curve = animation.drivers.find(path) if animation else None
        wants_driver = after or record['driver_present']
        if bool(curve) != wants_driver:
            raise RuntimeError('Grade dependency presence differs from its ownership receipt')
        if curve is not None:
            variables = curve.driver.variables
            if len(variables) != 1 or variables[0].targets[0].id != expected:
                raise RuntimeError('Grade dependency target differs from its ownership receipt')
            if not after and curve.mute != record['driver_mute']:
                raise RuntimeError('Grade dependency mute state differs from its ownership receipt')
        if after:
            if not expected.use_fake_user or snapshot(expected) != record['after_values']:
                raise RuntimeError('Created Grade differs from its ownership receipt')
            if record['created_action'] is not None:
                action = transactions.find_action(record['created_action'])
                old = transactions.find_action(record['old_action'])
                if action is None or action == old or expected.animation_data.action != action:
                    raise RuntimeError('Created Grade does not own its independent Action')
        elif _created_datablock(record) is not None or _created_datablock(record, action=True) is not None:
            raise RuntimeError('Failed Grade creation retained an allocated datablock')


    def ensure(node):
        return selected(node) or new(node, duplicate=False)


    def changed(value):
        """Commit a validated Grade edit or a node's local handle/numeric edit."""
        if core().recovery_pending():return
        if value.as_pointer() in _busy:
            return
        if is_data(value):
            if local(value):
                publish(value)
            return
        if not editable(value):
            sync_node(value)
            return
        grade = selected(value)
        if grade is None:
            # Selection-only clicks and no-op resets must not allocate a Grade.
            current, neutral = snapshot(value), defaults()
            def equal(a, b):
                if isinstance(a, dict):return all(equal(v, b[k]) for k, v in a.items())
                if isinstance(a, tuple):return all(equal(v, w) for v, w in zip(a, b))
                if isinstance(a, str):return a == b
                return abs(a-b) <= 2e-7*max(1., abs(a))
            if equal(current, neutral):
                core().refresh_node(value)
                return
            grade = ensure(value)
        else:
            assign_values(grade, snapshot(value))
        publish(grade)


    def selection_changed(node, context):
        if core().recovery_pending():return
        if node.as_pointer() in _busy or not core().is_grade(node):
            return
        if node.pg_grade is not None and not is_data(node.pg_grade):
            # PointerProperty.poll only filters the UI, not Python assignments.
            with guarded(node):
                node.pg_grade = None
        if selected(node) is None:
            assign_values(node, defaults())
        else:
            sync_node(node)
        bind_animation_dependency(node)
        core().refresh_node(node, force=True)
        redraw()


    def unlink(node):
        node.pg_grade = None


    def keyframe(node, family, name, group):
        grade = ensure(node)
        path = '["'+MATRIX+'"]' if family == 'pg_linear' else family+'.'+name
        grade.keyframe_insert(data_path=path, group=group)


    def draw_selector(layout, node):
        row = layout.row(align=True)
        row.enabled = local(node.id_data)
        row.context_pointer_set('pg_node', node)
        row.template_ID(node, 'pg_grade', new='perceptual_grade.new_grade',
                        unlink='perceptual_grade.unlink_grade')
        return row


    def draw_evaluation(layout,node):
        if linear.control_error(matrix(node),space(node))>2e-8:
            layout.label(text=tr('grade.matrix.readout'),icon='INFO')
        lut = _application.component('graph.lut')
        lut.draw_error(layout,node)


    class PGGradeData(bpy.types.NodeTree):
        bl_idname = 'PGGradeData'
        bl_label = native('grade.data.label')
        bl_icon = 'COLOR'

        @classmethod
        def poll(cls, context):
            return False  # Parameter storage; no additional node editor type.


    class GradeNode:
        bl_label = native('node.grade.title')
        bl_description = native('tooltip.action.add')

        def init(self, context):
            if not _converting:
                # Blender 5.2 adds a second ID user when node_tree is assigned
                # inside init(). The Add/Initialize operator finishes after return.
                self[core().TAG] = core().VERSION
                self[NODE_SCHEMA] = SCHEMA
                self['_pg_pending_init'] = True

        def copy(self, original):
            core().copy_output_resources(self, original)

        def free(self):
            release_node_references(self)

        def draw_buttons(self, context, layout):
            if core().recovery_pending():return
            if draw_resource_action(layout, self):return
            editing = _application.component('graph.editing')
            row = draw_selector(layout, self)
            row.operator('perceptual_grade.toggle_diagram', text='', icon='GREASEPENCIL',
                         depress=editing.shown(context, self))
            _application.component('graph.bake').draw_action(layout, self, context)
            lut = _application.component('graph.lut')
            lut.draw_error(layout,self)

        def draw_buttons_ext(self, context, layout):
            if core().recovery_pending():return
            if draw_resource_action(layout, self):return
            draw_selector(layout, self)
            _application.component('graph.bake').draw_action(layout, self, context)
            draw_evaluation(layout,self)


    class PGShaderGrade(GradeNode, bpy.types.ShaderNodeCustomGroup):
        bl_idname = 'PGShaderGrade'


    class PGCompositorGrade(GradeNode, bpy.types.CompositorNodeCustomGroup):
        bl_idname = 'PGCompositorGrade'


    def context_node(context):
        node = getattr(context, 'pg_node', None)
        node = node if core().is_grade(node) else core().active_grade(context)
        return node if core().is_grade(node) else None


    def resource_context_node(context):
        node = getattr(context, 'pg_node', None) or getattr(context, 'pg_stack_node', None)
        if node is None:
            space = context.space_data
            tree = space.edit_tree if space and space.type == 'NODE_EDITOR' else None
            node = tree.nodes.active if tree else None
        return node


    def draw_resource_action(layout, node):
        action = core().resource_action(node)
        if action is None:return False
        _application.component('integration.outcomes').draw(layout, node, 'resources')
        if action == 'invalid':
            layout.label(text=tr('output.resources.invalid'), icon='ERROR')
            return True
        status, label = {
            'initialize': ('output.resources.initialize.status', 'output.resources.initialize.label'),
            'independent': ('output.resources.independent.status', 'output.resources.independent.label'),
        }[action]
        layout.context_pointer_set('pg_node', node)
        layout.label(text=tr(status), icon='INFO')
        layout.operator('perceptual_grade.prepare_output',
                        text=tr(label))
        return True


    class PG_OT_prepare_output(bpy.types.Operator):
        bl_idname = 'perceptual_grade.prepare_output'
        bl_label = native('output.resources.prepare.label')
        bl_description = native('output.resources.prepare.tooltip')
        bl_translation_context = CONTEXT
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            output = _application.component('graph.output')
            return (not output.busy() and not output.native_running()
                    and core().resource_action(resource_context_node(context)) in ('initialize', 'independent'))

        def execute(self, context):
            node = resource_context_node(context)
            before = node.node_tree.as_pointer() if node.node_tree else None
            try:
                core().prepare_output_resources(node)
            except Exception as exc:
                _application.component('integration.outcomes').failure(node, 'resources', exc)
                # A later refresh failure cannot suppress the undo snapshot of a
                # completed structural installation. Failed compensation likewise
                # must retain an ordinary undoable result when a new edge remains.
                after = node.node_tree.as_pointer() if node.node_tree else None
                return {'FINISHED'} if after != before else {'CANCELLED'}
            redraw()
            return {'FINISHED'}


    class PG_OT_new_grade(bpy.types.Operator):
        bl_idname = 'perceptual_grade.new_grade'
        bl_label = native('grade.new.label')
        bl_description = native('grade.new.tooltip')
        bl_translation_context = CONTEXT
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            node = context_node(context)
            return node is not None and local(node.id_data)

        def execute(self, context):
            node = context_node(context)
            new(node)
            core().refresh_node(node, force=True)
            redraw()
            return {'FINISHED'}


    class PG_OT_unlink_grade(bpy.types.Operator):
        bl_idname = 'perceptual_grade.unlink_grade'
        bl_label = native('grade.unlink.label')
        bl_description = native('grade.unlink.tooltip')
        bl_translation_context = CONTEXT
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return PG_OT_new_grade.poll(context)

        def execute(self, context):
            unlink(context_node(context))
            return {'FINISHED'}


    def _drop_links(tree, node):
        for link in tuple(tree.links):
            if link.from_node == node or link.to_node == node:
                tree.links.remove(link)


    def _replacement_links(tree, node, original_name, links):
        """Install a saved edge set, raising instead of accepting a partial graph."""
        for src_name, src_id, src_label, dst_name, dst_id, dst_label in links:
            src = node if src_name == original_name else tree.nodes.get(src_name)
            dst = node if dst_name == original_name else tree.nodes.get(dst_name)
            if src is None or dst is None:
                raise RuntimeError('Grade conversion lost a connected node')
            output = next((s for s in src.outputs if s.identifier == src_id), src.outputs.get(src_label))
            input_ = next((s for s in dst.inputs if s.identifier == dst_id), dst.inputs.get(dst_label))
            if output is None or input_ is None:
                raise RuntimeError('Grade conversion lost a connected socket')
            tree.links.new(output, input_)


    def replace_node(node, kind, *, portable_state=None):
        """Prepare a portable/native replacement before retiring the complete node.

        RNA pointers, animation paths and external links are reversible until the
        old node is removed. On failure the old graph and its owned Grade users
        are restored; no partially initialized replacement is retained.
        """
        transactions = _application.component('ownership.transactions')
        n = core()
        tree = node.id_data
        if tree.is_evaluated or not local(tree):
            raise ValueError('Cannot replace a borrowed or linked Grade node')
        name, old_path = node.name, node.path_from_id()
        active = tree.nodes.active == node
        links = [(l.from_node.name, l.from_socket.identifier, l.from_socket.name,
                  l.to_node.name, l.to_socket.identifier, l.to_socket.name)
                 for l in tree.links if l.from_node == node or l.to_node == node]
        values = {s.identifier: tuple(s.default_value) if hasattr(s.default_value, '__len__')
                  and not isinstance(s.default_value, str) else s.default_value
                  for s in node.inputs if hasattr(s, 'default_value')}
        # Eagerly copy PropertyGroups while their owner is guaranteed to exist.
        properties = {key: value.to_dict() if hasattr(value, 'to_dict') else value
                      for key, value in node.items() if key not in
                      (transactions.NODE_ID, transactions.RECORDS, 'pg_bake_image')}
        settings, grade = snapshot(node), selected(node)
        bake = _application.component('graph.bake')
        bake_image = bake.baked_image(node)
        stacks = _application.component('graph.stacks'); editing = _application.component('graph.editing')
        stack_values = stacks.entries(node) if stacks.is_stack(node) else None
        focus = {key: getattr(node.pg_chromatic, key) for key in FOCUS_FIELDS}
        attributes = {key: getattr(node, key) for key in
                      ('label', 'width', 'hide', 'mute', 'select', 'use_custom_color', 'parent')}
        attributes.update(location=tuple(node.location), color=tuple(node.color))
        curves = [(curve, curve.data_path) for curve in owner_curves(tree)
                  if curve.data_path.startswith(old_path)]
        record = {'node': transactions.node_key(node), 'replacement': None, 'name': name,
                  'existing_nodes': tuple(transactions.node_key(item) for item in tree.nodes),
                  'replacement_kind': kind, 'creation_resolved': False,
                  'grade': transactions.id_key(grade),
                  'bake_image': transactions.id_key(bake_image),
                  'kernel': transactions.id_key(node.node_tree), 'settings': settings,
                  'stack': None if stack_values is None else transactions.stack_values(stack_values),
                  'links': links,
                  'socket_visibility': tuple((direction, socket.identifier, bool(socket.hide))
                      for direction, sockets in (('INPUT', node.inputs), ('OUTPUT', node.outputs))
                      for socket in sockets),
                  'curves': [(index, curve.data_path, curve.array_index)
                             for index, curve in enumerate(owner_curves(tree))],
                  'active': transactions.node_key(tree.nodes.active) if tree.nodes.active else None}
        tx = transactions.Transaction('replacement', record)
        replacement = None
        try:
            replacement = tx.apply('replacement_create', lambda: tree.nodes.new(kind),
                                   lambda: _discard_replacement(tx.record), retirement=True)
            tx.record['replacement'] = transactions.node_key(replacement)
            tx.record['creation_resolved'] = True
            tx.apply('replacement_kernel', lambda: setattr(replacement, 'node_tree', node.node_tree),
                     lambda: _clear_replacement_kernel(tx.record),
                     after_edges=(transactions.id_key(node.node_tree),))
            for key, value in properties.items():
                replacement[key] = value
            if portable_state is not None:
                replacement[WRAPPER_STATE] = portable_state
            elif kind in (*KINDS.values(), *stacks.KINDS.values()) and WRAPPER_STATE in replacement:
                del replacement[WRAPPER_STATE]
            assign_values(replacement, settings)
            def assign_grade():
                with guarded(replacement): replacement.pg_grade = grade
            tx.apply('replacement_grade', assign_grade, lambda: _clear_replacement_grade(tx.record),
                     after_edges=(transactions.id_key(grade),) if grade is not None else ())
            def assign_bake_image():
                with bake.binding_suppressed(replacement):
                    replacement.pg_bake_image = bake_image
            tx.apply('replacement_bake_image', assign_bake_image,
                     lambda: _clear_replacement_bake_image(tx.record),
                     after_edges=(transactions.id_key(bake_image),) if bake_image is not None else ())
            with guarded(replacement):
                for key, value in focus.items():
                    setattr(replacement.pg_chromatic, key, value)
            if stack_values is not None:
                tx.apply('replacement_stack', lambda: stacks._write_entries(replacement, stack_values),
                         lambda: _clear_replacement_stack(tx.record),
                         after_edges=tuple(transactions.id_key(value[0]) for value in stack_values if value[0] is not None))
            for key, value in attributes.items():
                setattr(replacement, key, value)
            replacement.show_options = kind in (*KINDS.values(), *stacks.KINDS.values())
            for socket in replacement.inputs:
                if socket.identifier in values:
                    socket.default_value = values[socket.identifier]
            for direction, identifier, hidden in record['socket_visibility']:
                sockets = replacement.inputs if direction == 'INPUT' else replacement.outputs
                sockets[identifier].hide = hidden
            def replace_links():
                _drop_links(tree, node)
                _replacement_links(tree, replacement, name, links)
            tx.apply('links_replace', replace_links, lambda: _restore_replacement_links(tx.record))
            # Rename only after settings/defaults/links have all been constructed.
            def rename_and_remap():
                node.name = '_PG replaced'
                replacement.name = name
                for curve, path in curves:
                    curve.data_path = replacement.path_from_id() + path[len(old_path):]
            tx.apply('names_and_animation_paths_replace', rename_and_remap,
                     lambda: _restore_replacement_identity(tx.record))
            if active:
                tx.apply('active_assign', lambda: setattr(tree.nodes, 'active', replacement),
                         lambda: _restore_replacement_active(tx.record))
            tx.apply('editor_rebind', lambda: editing.rebind(node, replacement),
                     lambda: _restore_replacement_editor(tx.record))
            owned = (transactions.id_key(grade),) if grade is not None else ()
            if bake_image is not None:owned += (transactions.id_key(bake_image),)
            if stack_values is not None:
                owned += tuple(transactions.id_key(value[0]) for value in stack_values if value[0] is not None)
            tx.apply('pointer_release', lambda: release_node_references(node),
                     lambda: _restore_replacement_owned(tx.record), before_edges=owned)
            # No fallible construction or ID mutation follows this commit point.
            try:
                tx.check_generation()
                remove_node = _application.app.blender.native.remove_node
                remove_node(tree, node)
                tx.check_generation()
            except Exception:
                # A native callback can raise after removal. Do not dereference the
                # invalid old wrapper or roll a completed commit backwards.
                tx.check_generation()
                if transactions.find_node(tree, tx.record['node']) is None:
                    tx.commit()
                    return replacement
                raise
            tx.commit()
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()
            n._signatures.clear()
        return replacement


    def _replacement_nodes(record):
        transactions = _application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        if tree is None: return None, None, None
        replacement = transactions.find_node(tree, record['replacement'])
        if replacement is None and not record['creation_resolved']:
            known = set()
            for identity in record['existing_nodes']:
                original = transactions.find_node(tree, identity)
                if original is not None: known.add(original.as_pointer())
            candidates = [item for item in tree.nodes if item.as_pointer() not in known]
            if len(candidates) > 1 or (candidates and candidates[0].bl_idname != record['replacement_kind']):
                raise RuntimeError('Native node creation has ambiguous partial effects')
            if candidates:
                replacement = candidates[0]
                record['replacement'] = transactions.node_key(replacement)
            record['creation_resolved'] = True
        return tree, transactions.find_node(tree, record['node']), replacement


    def _restore_replacement_owned(record):
        transactions = _application.component('ownership.transactions')
        stacks = _application.component('graph.stacks')
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is None: raise RuntimeError('Grade rollback lost its original node')
        grade = None if record['grade'] is None else transactions.require_tree(record['grade'])
        with guarded(node): node.pg_grade = grade
        bake = _application.component('graph.bake')
        image = transactions.find_id(record['bake_image'], bpy.data.images)
        if record['bake_image'] is not None and image is None:
            raise ReferenceError('Grade repair lost its baked Image')
        with bake.binding_suppressed(node):node.pg_bake_image = image
        if record['stack'] is not None:
            stacks._write_entries(node, transactions.restore_stack_values(record['stack']))


    def _restore_replacement_editor(record):
        editing = _application.component('graph.editing')
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is None: raise RuntimeError('Grade rollback lost its original node')
        if replacement is not None: editing.rebind(replacement, node)


    def _restore_replacement_active(record):
        transactions = _application.component('ownership.transactions')
        tree, node, replacement = _replacement_nodes(record)
        if tree is not None: tree.nodes.active = transactions.find_node(tree, record['active'])


    def _restore_replacement_curves(record):
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        curves = list(owner_curves(tree))
        for index, path, array_index in record['curves']:
            if index >= len(curves) or curves[index].array_index != array_index:
                raise RuntimeError('Grade rollback lost an animation curve')
            curves[index].data_path = path


    def _restore_replacement_names(record):
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is None: raise RuntimeError('Grade rollback lost its original node')
        if replacement is not None: replacement.name = '_PG failed replacement'
        node.name = record['name']


    def _restore_replacement_identity(record):
        # Native renaming remaps FCurve paths itself. Explicit paths must therefore
        # be restored AFTER both names, before the disposable node is removed.
        try: _restore_replacement_names(record)
        finally: _restore_replacement_curves(record)


    def _restore_replacement_links(record):
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is None: raise RuntimeError('Grade rollback lost its original node')
        if replacement is not None: _drop_links(tree, replacement)
        _drop_links(tree, node)
        _replacement_links(tree, node, record['name'], record['links'])


    def _discard_replacement(record):
        tree, node, replacement = _replacement_nodes(record)
        if tree is None or replacement is None: return
        release_node_references(replacement)
        remove_node = _application.app.blender.native.remove_node
        remove_node(tree, replacement)


    def _clear_replacement_kernel(record):
        tree, node, replacement = _replacement_nodes(record)
        if replacement is not None: replacement.node_tree = None


    def _clear_replacement_grade(record):
        tree, node, replacement = _replacement_nodes(record)
        if replacement is not None:
            with guarded(replacement): replacement.pg_grade = None


    def _clear_replacement_bake_image(record):
        tree, node, replacement = _replacement_nodes(record)
        if replacement is not None:
            with _application.component('graph.bake').binding_suppressed(replacement):
                replacement.pg_bake_image = None


    def _clear_replacement_stack(record):
        tree, node, replacement = _replacement_nodes(record)
        if replacement is not None:
            with guarded(replacement): replacement.pg_stack_entries.clear()


    def repair_replacement(record):
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is None:
            verify_replacement(record)
            return  # Native removal committed before its durable completion receipt.
        # Keep trying independent effects after one compensation fails. No caught
        # exception enters a list or survives this synchronous repair invocation.
        failed = False
        for restore in (_restore_replacement_owned, _restore_replacement_editor,
                        _restore_replacement_active, _restore_replacement_identity,
                        _restore_replacement_links):
            try: restore(record)
            except Exception: failed = True
        if failed: raise RuntimeError('Grade conversion ownership repair remains incomplete')
        _discard_replacement(record)
        _verify_replacement_contents(record, node)


    def verify_replacement(record):
        transactions = _application.component('ownership.transactions')
        stacks = _application.component('graph.stacks')
        tree, node, replacement = _replacement_nodes(record)
        if tree is None: return
        if node is not None or replacement is None:
            raise RuntimeError('Grade conversion has not completed native retirement')
        _verify_replacement_contents(record, replacement)


    def _verify_replacement_contents(record, target):
        transactions = _application.component('ownership.transactions')
        stacks = _application.component('graph.stacks')
        tree = target.id_data
        if target.name != record['name'] or target.node_tree != transactions.require_tree(record['kernel']):
            raise RuntimeError('Grade conversion kernel or identity does not match its receipt')
        grade = None if record['grade'] is None else transactions.require_tree(record['grade'])
        if selected(target) != grade or snapshot(target) != record['settings']:
            raise RuntimeError('Grade conversion settings or owned reference differ from its receipt')
        image = transactions.find_id(record['bake_image'], bpy.data.images)
        if (record['bake_image'] is not None and image is None) or target.pg_bake_image != image:
            raise RuntimeError('Grade conversion baked Image differs from its receipt')
        visibility = tuple((direction, socket.identifier, bool(socket.hide))
            for direction, sockets in (('INPUT', target.inputs), ('OUTPUT', target.outputs))
            for socket in sockets)
        if visibility != tuple(record['socket_visibility']):
            raise RuntimeError('Grade conversion socket visibility differs from its receipt')
        if record['stack'] is not None and stacks.entries(target) != transactions.restore_stack_values(record['stack']):
            raise RuntimeError('Grade conversion Stack occurrences differ from its receipt')
        links = [(link.from_node.name, link.from_socket.identifier, link.from_socket.name,
                  link.to_node.name, link.to_socket.identifier, link.to_socket.name)
                 for link in tree.links if link.from_node == target or link.to_node == target]
        if sorted(links) != sorted(record['links']):
            raise RuntimeError('Grade conversion links differ from its receipt')
        curves = [(curve.data_path, curve.array_index) for curve in owner_curves(tree)]
        if sorted(curves) != sorted((path, index) for _, path, index in record['curves']):
            raise RuntimeError('Grade conversion animation differs from its receipt')


    def wrapper_plan(node):
        """Read only the explicit current-format wrapper serialized by this product."""
        if node.bl_idname not in ('ShaderNodeGroup', 'CompositorNodeGroup'):
            return None
        encoded = node.get(WRAPPER_STATE)
        if encoded is None:
            return None
        stacks = _application.component('graph.stacks')
        try:
            if type(encoded) is not str or len(encoded) > 512:
                raise ValueError('Malformed wrapper metadata encoding')
            state = json.loads(encoded)
            kind = (stacks.KINDS if stacks.is_stack(node) else KINDS)[node.id_data.bl_idname]
            if (type(state) is not dict or type(state.get('version')) is not int or state['version'] != 1
                    or state.get('kind') != kind or 'driver_mute' not in state
                    or state['driver_mute'] is not None and type(state['driver_mute']) is not bool):
                raise ValueError('Unsupported saved control wrapper metadata')
        except (TypeError, ValueError, KeyError) as error:
            raise RuntimeError('Saved control wrapper metadata is invalid') from error
        return kind, state.get('driver_mute')


    def restore_wrapper(node):
        """Restore one saved UI wrapper, retaining its adopted native output."""
        plan = wrapper_plan(node)
        if plan is None:
            return node
        lut = _application.component('graph.lut')
        if not lut.is_adopted(node):
            raise RuntimeError('Saved control restoration requires adopted native output')
        identity = (int(node.id_data.session_uid), node.name)
        tree, name = node.id_data, node.name
        with _application.component('graph.output_guard').publication():
            representations(restore_adopted=True, restore_plan={identity: plan}, owners=(node,))
        return tree.nodes[name]


    def representations(portable=False, *, restore_adopted=False, restore_plan=None, owners=None):
        nonlocal _converting
        if _converting or not hasattr(bpy.data, 'node_groups'):
            return
        _converting = True
        try:
            stacks = _application.component('graph.stacks'); lut = _application.component('graph.lut')
            transactions = _application.component('ownership.transactions')
            for node in list(core().all_nodes() if owners is None else owners):
                if not local(node.id_data):
                    continue
                adopted = lut.is_adopted(node)
                identity=(int(node.id_data.session_uid),node.name)
                if restore_plan is not None and identity not in restore_plan:
                    continue
                if not portable and not restore_adopted and node.bl_idname in ('ShaderNodeGroup','CompositorNodeGroup'):
                    # Native wrappers remain stable after adoption, including
                    # when a later control edit resumes LUT production.
                    continue
                if adopted and not (portable or restore_adopted):
                    # Recovered native nodes, links and driver settings are
                    # authored data. Idle/save restoration cannot reconstruct
                    # them merely because the new Session has no producer.
                    continue
                kind = ('ShaderNodeGroup' if node.id_data.bl_idname == 'ShaderNodeTree'
                        else 'CompositorNodeGroup') if portable else (stacks.KINDS if stacks.is_stack(node) else KINDS)[node.id_data.bl_idname]
                if restore_plan is not None:kind=restore_plan[identity][0]
                animation = node.id_data.animation_data
                curve = None
                if animation is not None:
                    path = node.path_from_id()+('["_pg_stack_dependency"]' if stacks.is_stack(node) else '["_pg_grade_dependency"]')
                    curve = animation.drivers.find(path)
                desired_mute=portable if restore_plan is None else restore_plan[identity][1]
                portable_state = None
                if portable and node.bl_idname in (*KINDS.values(), *stacks.KINDS.values()):
                    portable_state = json.dumps(dict(version=1, kind=node.bl_idname,
                        driver_mute=None if curve is None else bool(curve.mute)), sort_keys=True)
                record = {'tree': transactions.id_key(node.id_data), 'path': curve.data_path if curve else '',
                          'index': curve.array_index if curve else 0, 'mute': curve.mute if curve else False,
                          'after_mute': desired_mute}
                tx = transactions.Transaction('driver', record)
                replacement, completed = None, None
                try:
                    if curve is not None and desired_mute is not None and curve.mute != desired_mute:
                        tx.apply('driver_mute', lambda: setattr(curve, 'mute', desired_mute),
                                 lambda: transactions.repair_driver(record))
                    if node.bl_idname != kind:
                        if not adopted:
                            try:
                                completed = lut.validate_output(node)
                            except (ValueError, TypeError, KeyError, RuntimeError, ReferenceError):
                                pass
                        replacement = replace_node(node, kind, portable_state=portable_state)
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise
                finally:
                    tx.close()
                if replacement is not None and adopted:
                    # Explicit portable save/restore only replaces the wrapper.
                    # Its preserved native output remains adopted in the new
                    # owner, without inspection or reconstructed provenance.
                    lut.adopt_native(replacement, core().output_intent(replacement))
                elif replacement is not None and completed is not None:
                    # The kernel survives conversion, but the graph owner does
                    # not. Adopt the preserved native graph without numerical claims.
                    lut.verify_replacement(replacement, completed)
        finally:
            _converting = False


    def resume_after_save():
        """Idempotent fallback when Blender omits save_post/save_post_fail."""
        if _saving:
            core().save_finished_handler()
            if _saving:
                return .1
        return None


    def prepare_save():
        nonlocal _saving
        scheduler = _application.component('graph.scheduler')
        _save_wrappers.clear()
        stacks=_application.component('graph.stacks')
        for node in core().all_nodes():
            if not local(node.id_data):continue
            wrapper_plan(node)  # Absence is historical; a present marker must validate.
            animation=node.id_data.animation_data
            path=node.path_from_id()+('["_pg_stack_dependency"]' if stacks.is_stack(node) else '["_pg_grade_dependency"]')
            curve=None if animation is None else animation.drivers.find(path)
            _save_wrappers[int(node.id_data.session_uid),node.name]=(
                node.bl_idname,None if curve is None else bool(curve.mute))
        _saving = True
        try:
            core().cube_nodes.clear()
            representations(portable=True)
            # The next application-timer turn is after synchronous serialization.
            # Completion normally removes this fallback before it can run.
            scheduler.wake(resume_after_save, delay=5.)
        except Exception:
            _saving = False
            scheduler.unwake(resume_after_save)
            try:
                representations(restore_adopted=True,restore_plan=_save_wrappers)
            except Exception:
                log_exception(LOG, 'Could not finish rollback of interrupted save preparation')
            raise


    def finish_save():
        nonlocal _saving
        scheduler = _application.component('graph.scheduler')
        _saving = False
        scheduler.unwake(resume_after_save)
        try:
            representations(restore_adopted=True,restore_plan=_save_wrappers)
        finally:
            # Conversion failure cannot leave refresh permanently suppressed.
            recover_flags()
        core().refresh_all(force=True)


    def recover_flags():
        """Clear only transient critical sections; keep registered RNA classes."""
        nonlocal _saving, _converting
        _busy.clear()
        _save_wrappers.clear()
        _saving = _converting = False


    def prepare_disable():
        if hasattr(bpy.data, 'node_groups'):
            output_guard = _application.component('graph.output_guard'); output_resources = _application.component('graph.output_resources')
            output_guard.require_available()
            packed = set()
            for node in core().all_nodes():
                if core().cube_nodes.is_adopted(node):
                    continue
                if not output_resources.is_current(node):
                    continue
                pointer = node.node_tree.as_pointer()
                if pointer not in packed:
                    # Preserve already installed bytes without computing new
                    # output or manufacturing completion evidence on disable.
                    output_resources.pack_output(node)
                    packed.add(pointer)
            core().cube_nodes.clear()
            representations(portable=True)


    def register():
        n = core()
        outcomes = _application.component('integration.outcomes')
        for cls in (PGGradeData, PGShaderGrade, PGCompositorGrade, PG_OT_new_grade,
                    PG_OT_unlink_grade, PG_OT_prepare_output):
            _application.register_class(cls)
            _classes.append(cls)
        for name, cls in (('pg_linear', n.PGLinearSettings), ('pg_tone', n.PGToneSettings),
                          ('pg_chromatic', n.PGChromaticSettings)):
            setattr(bpy.types.NodeTree, name, PointerProperty(type=cls))
        labels = ('symbol.d65', 'white.d60', 'symbol.aces', 'symbol.d50', 'symbol.d55',
                  'symbol.d75', 'white.dci', 'white.e')
        for cls in (bpy.types.NodeTree, bpy.types.Node):
            cls.pg_working_white = outcomes.property(EnumProperty(name=native('grade.white.label'),
                description=native('grade.white.tooltip'), translation_context=CONTEXT,
                items=[(key, native(label), '', i) for i, (key, label) in enumerate(zip(WHITES, labels))],
                get=white_get, set=white_set, options=set()), control='white')
        bpy.types.Node.pg_grade = outcomes.property(PointerProperty(type=bpy.types.NodeTree,
            name=native('grade.data.label'), description=native('grade.selector.tooltip'),
            poll=lambda node, value: is_data(value), update=selection_changed), control='grade')


    def clear():
        nonlocal _saving, _converting
        recover_flags()
        _classes.clear()


    def unregister():
        if hasattr(bpy.types.Node, 'pg_grade'):
            del bpy.types.Node.pg_grade
        for family in ('pg_linear', *(name for name, _ in FAMILIES)):
            if hasattr(bpy.types.NodeTree, family):
                delattr(bpy.types.NodeTree, family)
        for cls in (bpy.types.NodeTree, bpy.types.Node):
            if hasattr(cls, 'pg_working_white'):del cls.pg_working_white
        for cls in reversed(_classes):
            _application.unregister_class(cls)
        clear()

    def _set_export_FAMILIES(value):
        nonlocal FAMILIES
        FAMILIES = value

    def _set_export_FOCUS_FIELDS(value):
        nonlocal FOCUS_FIELDS
        FOCUS_FIELDS = value

    def _set_export_KINDS(value):
        nonlocal KINDS
        KINDS = value

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export_MATRIX(value):
        nonlocal MATRIX
        MATRIX = value

    def _set_export_MATRIX_WHITE(value):
        nonlocal MATRIX_WHITE
        MATRIX_WHITE = value

    def _set_export_NODE_SCHEMA(value):
        nonlocal NODE_SCHEMA
        NODE_SCHEMA = value

    def _set_export_SCHEMA(value):
        nonlocal SCHEMA
        SCHEMA = value

    def _set_export_TAG(value):
        nonlocal TAG
        TAG = value

    def _set_export_WHITE(value):
        nonlocal WHITE
        WHITE = value

    def _set_export__busy(value):
        nonlocal _busy
        _busy = value

    def _set_export__classes(value):
        nonlocal _classes
        _classes = value

    def _set_export__converting(value):
        nonlocal _converting
        _converting = value

    def _set_export__saving(value):
        nonlocal _saving
        _saving = value

    def _set_export_core(value):
        nonlocal core
        core = value

    def _set_export_is_data(value):
        nonlocal is_data
        is_data = value

    def _set_export_selected(value):
        nonlocal selected
        selected = value

    def _set_export_data_owner(value):
        nonlocal data_owner
        data_owner = value

    def _set_export_local(value):
        nonlocal local
        local = value

    def _set_export_editable(value):
        nonlocal editable
        editable = value

    def _set_export_guarded(value):
        nonlocal guarded
        guarded = value

    def _set_export_release_node_references(value):
        nonlocal release_node_references
        release_node_references = value

    def _set_export_snapshot(value):
        nonlocal snapshot
        snapshot = value

    def _set_export_assign_values(value):
        nonlocal assign_values
        assign_values = value

    def _set_export_defaults(value):
        nonlocal defaults
        defaults = value

    def _set_export_white(value):
        nonlocal white
        white = value

    def _set_export_space(value):
        nonlocal space
        space = value

    def _set_export_matrix(value):
        nonlocal matrix
        matrix = value

    def _set_export_write_matrix(value):
        nonlocal write_matrix
        write_matrix = value

    def _set_export_parameters(value):
        nonlocal parameters
        parameters = value

    def _set_export_edit_linear(value):
        nonlocal edit_linear
        edit_linear = value

    def _set_export_white_get(value):
        nonlocal white_get
        white_get = value

    def _set_export_white_set(value):
        nonlocal white_set
        white_set = value

    def _set_export_users(value):
        nonlocal users
        users = value

    def _set_export_action_curves(value):
        nonlocal action_curves
        action_curves = value

    def _set_export_owner_curves(value):
        nonlocal owner_curves
        owner_curves = value

    def _set_export_sync_node(value):
        nonlocal sync_node
        sync_node = value

    def _set_export_bind_animation_dependency(value):
        nonlocal bind_animation_dependency
        bind_animation_dependency = value

    def _set_export_redraw(value):
        nonlocal redraw
        redraw = value

    def _set_export_publish(value):
        nonlocal publish
        publish = value

    def _set_export_new(value):
        nonlocal new
        new = value

    def _set_export__created_datablock(value):
        nonlocal _created_datablock
        _created_datablock = value

    def _set_export__creation_node(value):
        nonlocal _creation_node
        _creation_node = value

    def _set_export__restore_grade_selector(value):
        nonlocal _restore_grade_selector
        _restore_grade_selector = value

    def _set_export__restore_created_action_link(value):
        nonlocal _restore_created_action_link
        _restore_created_action_link = value

    def _set_export__discard_created_grade(value):
        nonlocal _discard_created_grade
        _discard_created_grade = value

    def _set_export__discard_created_action(value):
        nonlocal _discard_created_action
        _discard_created_action = value

    def _set_export_repair_grade_creation(value):
        nonlocal repair_grade_creation
        repair_grade_creation = value

    def _set_export_verify_grade_creation(value):
        nonlocal verify_grade_creation
        verify_grade_creation = value

    def _set_export_ensure(value):
        nonlocal ensure
        ensure = value

    def _set_export_changed(value):
        nonlocal changed
        changed = value

    def _set_export_selection_changed(value):
        nonlocal selection_changed
        selection_changed = value

    def _set_export_unlink(value):
        nonlocal unlink
        unlink = value

    def _set_export_keyframe(value):
        nonlocal keyframe
        keyframe = value

    def _set_export_draw_selector(value):
        nonlocal draw_selector
        draw_selector = value

    def _set_export_draw_evaluation(value):
        nonlocal draw_evaluation
        draw_evaluation = value

    def _set_export_PGGradeData(value):
        nonlocal PGGradeData
        PGGradeData = value

    def _set_export_GradeNode(value):
        nonlocal GradeNode
        GradeNode = value

    def _set_export_PGShaderGrade(value):
        nonlocal PGShaderGrade
        PGShaderGrade = value

    def _set_export_PGCompositorGrade(value):
        nonlocal PGCompositorGrade
        PGCompositorGrade = value

    def _set_export_context_node(value):
        nonlocal context_node
        context_node = value

    def _set_export_resource_context_node(value):
        nonlocal resource_context_node
        resource_context_node = value

    def _set_export_draw_resource_action(value):
        nonlocal draw_resource_action
        draw_resource_action = value

    def _set_export_PG_OT_prepare_output(value):
        nonlocal PG_OT_prepare_output
        PG_OT_prepare_output = value

    def _set_export_PG_OT_new_grade(value):
        nonlocal PG_OT_new_grade
        PG_OT_new_grade = value

    def _set_export_PG_OT_unlink_grade(value):
        nonlocal PG_OT_unlink_grade
        PG_OT_unlink_grade = value

    def _set_export__drop_links(value):
        nonlocal _drop_links
        _drop_links = value

    def _set_export__replacement_links(value):
        nonlocal _replacement_links
        _replacement_links = value

    def _set_export_replace_node(value):
        nonlocal replace_node
        replace_node = value

    def _set_export__replacement_nodes(value):
        nonlocal _replacement_nodes
        _replacement_nodes = value

    def _set_export__restore_replacement_owned(value):
        nonlocal _restore_replacement_owned
        _restore_replacement_owned = value

    def _set_export__restore_replacement_editor(value):
        nonlocal _restore_replacement_editor
        _restore_replacement_editor = value

    def _set_export__restore_replacement_active(value):
        nonlocal _restore_replacement_active
        _restore_replacement_active = value

    def _set_export__restore_replacement_curves(value):
        nonlocal _restore_replacement_curves
        _restore_replacement_curves = value

    def _set_export__restore_replacement_names(value):
        nonlocal _restore_replacement_names
        _restore_replacement_names = value

    def _set_export__restore_replacement_identity(value):
        nonlocal _restore_replacement_identity
        _restore_replacement_identity = value

    def _set_export__restore_replacement_links(value):
        nonlocal _restore_replacement_links
        _restore_replacement_links = value

    def _set_export__discard_replacement(value):
        nonlocal _discard_replacement
        _discard_replacement = value

    def _set_export__clear_replacement_kernel(value):
        nonlocal _clear_replacement_kernel
        _clear_replacement_kernel = value

    def _set_export__clear_replacement_grade(value):
        nonlocal _clear_replacement_grade
        _clear_replacement_grade = value

    def _set_export__clear_replacement_stack(value):
        nonlocal _clear_replacement_stack
        _clear_replacement_stack = value

    def _set_export_repair_replacement(value):
        nonlocal repair_replacement
        repair_replacement = value

    def _set_export_verify_replacement(value):
        nonlocal verify_replacement
        verify_replacement = value

    def _set_export__verify_replacement_contents(value):
        nonlocal _verify_replacement_contents
        _verify_replacement_contents = value

    def _set_export_representations(value):
        nonlocal representations
        representations = value

    def _set_export_resume_after_save(value):
        nonlocal resume_after_save
        resume_after_save = value

    def _set_export_prepare_save(value):
        nonlocal prepare_save
        prepare_save = value

    def _set_export_finish_save(value):
        nonlocal finish_save
        finish_save = value

    def _set_export_recover_flags(value):
        nonlocal recover_flags
        recover_flags = value

    def _set_export_prepare_disable(value):
        nonlocal prepare_disable
        prepare_disable = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CONTEXT': lambda: CONTEXT,
        'EnumProperty': lambda: EnumProperty,
        'FAMILIES': lambda: FAMILIES,
        'FOCUS_FIELDS': lambda: FOCUS_FIELDS,
        'GradeNode': lambda: GradeNode,
        'KINDS': lambda: KINDS,
        'LOG': lambda: LOG,
        'MATRIX': lambda: MATRIX,
        'MATRIX_WHITE': lambda: MATRIX_WHITE,
        'NODE_SCHEMA': lambda: NODE_SCHEMA,
        'PGCompositorGrade': lambda: PGCompositorGrade,
        'PGGradeData': lambda: PGGradeData,
        'PGShaderGrade': lambda: PGShaderGrade,
        'PG_OT_new_grade': lambda: PG_OT_new_grade,
        'PG_OT_prepare_output': lambda: PG_OT_prepare_output,
        'PG_OT_unlink_grade': lambda: PG_OT_unlink_grade,
        'PointerProperty': lambda: PointerProperty,
        'SCHEMA': lambda: SCHEMA,
        'TAG': lambda: TAG,
        'WHITE': lambda: WHITE,
        'WRAPPER_STATE': lambda: WRAPPER_STATE,
        '_busy': lambda: _busy,
        '_classes': lambda: _classes,
        '_clear_replacement_grade': lambda: _clear_replacement_grade,
        '_clear_replacement_kernel': lambda: _clear_replacement_kernel,
        '_clear_replacement_stack': lambda: _clear_replacement_stack,
        '_converting': lambda: _converting,
        '_created_datablock': lambda: _created_datablock,
        '_creation_node': lambda: _creation_node,
        '_discard_created_action': lambda: _discard_created_action,
        '_discard_created_grade': lambda: _discard_created_grade,
        '_discard_replacement': lambda: _discard_replacement,
        '_drop_links': lambda: _drop_links,
        '_replacement_links': lambda: _replacement_links,
        '_replacement_nodes': lambda: _replacement_nodes,
        '_restore_created_action_link': lambda: _restore_created_action_link,
        '_restore_grade_selector': lambda: _restore_grade_selector,
        '_restore_replacement_active': lambda: _restore_replacement_active,
        '_restore_replacement_curves': lambda: _restore_replacement_curves,
        '_restore_replacement_editor': lambda: _restore_replacement_editor,
        '_restore_replacement_identity': lambda: _restore_replacement_identity,
        '_restore_replacement_links': lambda: _restore_replacement_links,
        '_restore_replacement_names': lambda: _restore_replacement_names,
        '_restore_replacement_owned': lambda: _restore_replacement_owned,
        '_saving': lambda: _saving,
        '_verify_replacement_contents': lambda: _verify_replacement_contents,
        'action_curves': lambda: action_curves,
        'assign_values': lambda: assign_values,
        'bind_animation_dependency': lambda: bind_animation_dependency,
        'bpy': lambda: bpy,
        'changed': lambda: changed,
        'clear': lambda: clear,
        'color': lambda: color,
        'context_node': lambda: context_node,
        'contextmanager': lambda: contextmanager,
        'core': lambda: core,
        'data_owner': lambda: data_owner,
        'defaults': lambda: defaults,
        'draw_evaluation': lambda: draw_evaluation,
        'draw_resource_action': lambda: draw_resource_action,
        'draw_selector': lambda: draw_selector,
        'edit_linear': lambda: edit_linear,
        'editable': lambda: editable,
        'ensure': lambda: ensure,
        'error_text': lambda: error_text,
        'finish_save': lambda: finish_save,
        'guarded': lambda: guarded,
        'is_data': lambda: is_data,
        'keyframe': lambda: keyframe,
        'linear': lambda: linear,
        'local': lambda: local,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'matrix': lambda: matrix,
        'native': lambda: native,
        'new': lambda: new,
        'owner_curves': lambda: owner_curves,
        'parameters': lambda: parameters,
        'prepare_disable': lambda: prepare_disable,
        'prepare_save': lambda: prepare_save,
        'publish': lambda: publish,
        'recover_flags': lambda: recover_flags,
        'wrapper_plan': lambda: wrapper_plan,
        'restore_wrapper': lambda: restore_wrapper,
        'redraw': lambda: redraw,
        'register': lambda: register,
        'release_node_references': lambda: release_node_references,
        'repair_grade_creation': lambda: repair_grade_creation,
        'repair_replacement': lambda: repair_replacement,
        'replace_node': lambda: replace_node,
        'representations': lambda: representations,
        'resource_context_node': lambda: resource_context_node,
        'resume_after_save': lambda: resume_after_save,
        'selected': lambda: selected,
        'selection_changed': lambda: selection_changed,
        'snapshot': lambda: snapshot,
        'space': lambda: space,
        'sync_node': lambda: sync_node,
        'tone': lambda: tone,
        'tr': lambda: tr,
        'unlink': lambda: unlink,
        'unregister': lambda: unregister,
        'users': lambda: users,
        'verify_grade_creation': lambda: verify_grade_creation,
        'verify_replacement': lambda: verify_replacement,
        'white': lambda: white,
        'white_get': lambda: white_get,
        'white_set': lambda: white_set,
        'write_matrix': lambda: write_matrix,
    }, writers={
        'FAMILIES': _set_export_FAMILIES,
        'FOCUS_FIELDS': _set_export_FOCUS_FIELDS,
        'KINDS': _set_export_KINDS,
        'LOG': _set_export_LOG,
        'MATRIX': _set_export_MATRIX,
        'MATRIX_WHITE': _set_export_MATRIX_WHITE,
        'NODE_SCHEMA': _set_export_NODE_SCHEMA,
        'SCHEMA': _set_export_SCHEMA,
        'TAG': _set_export_TAG,
        'WHITE': _set_export_WHITE,
        '_busy': _set_export__busy,
        '_classes': _set_export__classes,
        '_converting': _set_export__converting,
        '_saving': _set_export__saving,
    'core': _set_export_core, 'is_data': _set_export_is_data, 'selected': _set_export_selected, 'data_owner': _set_export_data_owner, 'local': _set_export_local, 'editable': _set_export_editable, 'guarded': _set_export_guarded, 'release_node_references': _set_export_release_node_references, 'snapshot': _set_export_snapshot, 'assign_values': _set_export_assign_values, 'defaults': _set_export_defaults, 'white': _set_export_white, 'space': _set_export_space, 'matrix': _set_export_matrix, 'write_matrix': _set_export_write_matrix, 'parameters': _set_export_parameters, 'edit_linear': _set_export_edit_linear, 'white_get': _set_export_white_get, 'white_set': _set_export_white_set, 'users': _set_export_users, 'action_curves': _set_export_action_curves, 'owner_curves': _set_export_owner_curves, 'sync_node': _set_export_sync_node, 'bind_animation_dependency': _set_export_bind_animation_dependency, 'redraw': _set_export_redraw, 'publish': _set_export_publish, 'new': _set_export_new, '_created_datablock': _set_export__created_datablock, '_creation_node': _set_export__creation_node, '_restore_grade_selector': _set_export__restore_grade_selector, '_restore_created_action_link': _set_export__restore_created_action_link, '_discard_created_grade': _set_export__discard_created_grade, '_discard_created_action': _set_export__discard_created_action, 'repair_grade_creation': _set_export_repair_grade_creation, 'verify_grade_creation': _set_export_verify_grade_creation, 'ensure': _set_export_ensure, 'changed': _set_export_changed, 'selection_changed': _set_export_selection_changed, 'unlink': _set_export_unlink, 'keyframe': _set_export_keyframe, 'draw_selector': _set_export_draw_selector, 'draw_evaluation': _set_export_draw_evaluation, 'PGGradeData': _set_export_PGGradeData, 'GradeNode': _set_export_GradeNode, 'PGShaderGrade': _set_export_PGShaderGrade, 'PGCompositorGrade': _set_export_PGCompositorGrade, 'context_node': _set_export_context_node, 'resource_context_node': _set_export_resource_context_node, 'draw_resource_action': _set_export_draw_resource_action, 'PG_OT_prepare_output': _set_export_PG_OT_prepare_output, 'PG_OT_new_grade': _set_export_PG_OT_new_grade, 'PG_OT_unlink_grade': _set_export_PG_OT_unlink_grade, '_drop_links': _set_export__drop_links, '_replacement_links': _set_export__replacement_links, 'replace_node': _set_export_replace_node, '_replacement_nodes': _set_export__replacement_nodes, '_restore_replacement_owned': _set_export__restore_replacement_owned, '_restore_replacement_editor': _set_export__restore_replacement_editor, '_restore_replacement_active': _set_export__restore_replacement_active, '_restore_replacement_curves': _set_export__restore_replacement_curves, '_restore_replacement_names': _set_export__restore_replacement_names, '_restore_replacement_identity': _set_export__restore_replacement_identity, '_restore_replacement_links': _set_export__restore_replacement_links, '_discard_replacement': _set_export__discard_replacement, '_clear_replacement_kernel': _set_export__clear_replacement_kernel, '_clear_replacement_grade': _set_export__clear_replacement_grade, '_clear_replacement_stack': _set_export__clear_replacement_stack, 'repair_replacement': _set_export_repair_replacement, 'verify_replacement': _set_export_verify_replacement, '_verify_replacement_contents': _set_export__verify_replacement_contents, 'representations': _set_export_representations, 'resume_after_save': _set_export_resume_after_save, 'prepare_save': _set_export_prepare_save, 'finish_save': _set_export_finish_save, 'recover_flags': _set_export_recover_flags, 'prepare_disable': _set_export_prepare_disable, 'register': _set_export_register, 'clear': _set_export_clear, 'unregister': _set_export_unregister})
