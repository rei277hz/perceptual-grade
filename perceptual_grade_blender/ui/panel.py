"""Native sidebar controls and actions; the chart is a node-editor overlay."""

from ..integration.components import Component

def create(_application):
    remove_node_group = _application.app.blender.native.remove_node_group; remove_node = _application.app.blender.native.remove_node
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text
    native = _application.component('localization.i18n').native
    tr = _application.component('localization.i18n').tr; tip = _application.component('localization.i18n').tip; error_text = _application.component('localization.i18n').error_text; CONTEXT = _application.component('localization.i18n').CONTEXT

    outcomes = _application.component('integration.outcomes')

    import bpy
    import math
    import numpy as np
    from functools import lru_cache
    from bpy.props import BoolProperty, FloatProperty, IntProperty, EnumProperty
    nodes = _application.component('graph.nodes')
    from ...perceptual_grade_core.model import linear as cm, tone as tm, color as ch
    diagram = _application.component('ui.diagram')
    scopes = _application.component('scopes.sampling')
    controls = _application.component('model.controls')

    RELEASE = '1.19.0'

    _keymaps = []
    _classes = []
    _menu_registered = False
    _diagram_registered = False



    def prepare_linear_properties():
        """Signed display views for the saturation-directed Balance controls."""
        def get_cr(self):return -self.cr_balance
        def set_cr(self,value):self.cr_balance=-value
        def get_yb(self):return -self.yb_balance
        def set_yb(self,value):self.yb_balance=-value
        for name,label,get,set_ in (
            ('cr_balance_display','control.cr_balance.label',get_cr,set_cr),
            ('yb_balance_display','control.yb_balance.label',get_yb,set_yb)):
            nodes.PGLinearSettings.__annotations__[name]=FloatProperty(translation_context=CONTEXT, 
                name=native(label),description=native('tooltip.linear.'+('cr_balance' if name.startswith('cr') else 'yb_balance')),
                default=0.,min=-1e12,max=1e12,soft_min=-2.,soft_max=2.,precision=3,get=get,set=set_,options=set())
        # Apply the same expanded editing ranges/precision used by diagram drags.
        for field,spec in controls.OVERRIDES.items():
            _,label,default,lo,hi,precision,_=spec
            nodes.PGChromaticSettings.__annotations__[field]=FloatProperty(translation_context=CONTEXT, 
                name=native(label),default=default,min=lo,max=hi,precision=precision,
                step=5 if field=='adaptance_half_saturation' else (10 if precision==1 else 1),
                update=nodes.changed,
                description=native(controls.tooltip_id(field)))
        prepare_warp_properties()


    @lru_cache(maxsize=16)
    def _warp_source_angles(tone_params,space):
        model=ch.model(tone_params,ch.Parameters(),space)
        return tuple(tuple(math.degrees(float(angle)) for angle in row)
                     for row in model.web.geometry.angles)


    def warp_hue_origin(node, tone, handle):
        """Source hue is independent of the current web's validity or Adaptance."""
        return _warp_source_angles(nodes.read_tone(node),nodes.grading_space(node))[tone][handle]


    def warp_hue_value(node, tone, handle):
        values=nodes.grades.data_owner(node).pg_chromatic
        return (warp_hue_origin(node,tone,handle)+getattr(values,f'web_hue_{tone}_{handle}'))%360.


    def warp_hue_delta(node, tone, handle, hue):
        """Choose the nearest stored lift of an absolute pre-Adaptance target hue."""
        field=f'web_hue_{tone}_{handle}'
        current=getattr(nodes.grades.data_owner(node).pg_chromatic,field)
        origin=warp_hue_origin(node,tone,handle)
        # RNA returns a binary32 display value. Clicking a field without changing
        # it must not create a tiny delta merely from that round trip.
        if np.float32(float(hue)%360.)==np.float32((origin+current)%360.):return current
        base=float(hue)%360.-origin
        lo,hi=controls.meta(field)[3:5]
        choices=[base+360.*turn for turn in range(-3,4) if lo<=base+360.*turn<=hi]
        return min(choices,key=lambda value:abs(value-current))


    def warp_sidebar_property(field):
        """Unsaved UI adapter; grading, animation and undo retain their raw paths."""
        hue=field.startswith('web_hue_')
        indices=tuple(map(int,field.rsplit('_',2)[1:])) if hue else None
        def get(settings):
            node=nodes.owner(settings)
            if hue and node is not None:return warp_hue_value(node,*indices)
            return getattr(settings,field)
        def set_(settings,value):
            node=nodes.owner(settings)
            if node is None:return
            if hue:value=warp_hue_delta(node,*indices,value)
            warp_edit = _application.component('ui.warp_edit')
            warp_edit.edit_sidebar(node,field,value)
        _,label,default,lo,hi,precision,_=controls.meta(field)
        args=dict(name=native(label),translation_context=CONTEXT,default=default,
                  min=lo,max=hi,precision=precision,step=1,get=get,set=set_,options={'SKIP_SAVE'},
                  description=native(controls.tooltip_id(field)))
        if hue:args.update(min=0.,max=360.,description=native('tooltip.warp.absolute_hue'))
        if field.startswith('web_scale_'):args.update(soft_min=-1.,soft_max=1.,precision=2)
        if field=='adaptance_half_saturation':args.update(step=5)
        return FloatProperty(**args)


    def prepare_warp_properties():
        for field in color_fields('WARP'):
            nodes.PGChromaticSettings.__annotations__['sidebar_'+field]=warp_sidebar_property(field)


    def supported_editor(context):
        return (context.space_data and context.space_data.type == 'NODE_EDITOR' and
                context.space_data.tree_type in {'ShaderNodeTree', 'CompositorNodeTree'})


    def redraw(context):
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()


    class AddGradeOperator:
        bl_idname = 'perceptual_grade.add_linear'
        bl_label = native('product.name')
        bl_translation_context = CONTEXT
        bl_description = native('tooltip.action.add')
        bl_options = {'REGISTER', 'UNDO'}
        stack = False

        @classmethod
        def poll(cls, context):
            return supported_editor(context)

        def execute(self, context):
            space = context.space_data
            created_tree = None
            tree = None
            existing_nodes = None
            try:
                nodes.working_space()
                tree = space.edit_tree
                new_tree = False
                if tree is None and space.tree_type == 'CompositorNodeTree':
                    tree = bpy.data.node_groups.new(tr('datablock.compositing'), 'CompositorNodeTree')
                    created_tree = tree
                    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
                    context.scene.compositing_node_group = tree
                    new_tree = True
                if tree is None:
                    outcomes.set(None, 'add', 'warning.select_shader_tree')
                    return {'CANCELLED'}
                if tree.library and not tree.override_library:
                    outcomes.set(None, 'add', 'warning.local_tree')
                    return {'CANCELLED'}
                existing_nodes = {item.as_pointer() for item in tree.nodes}
                stacks = _application.component('graph.stacks')
                node = stacks.create(tree) if self.stack else nodes.create_grade(tree)
                node.location = space.cursor_location
                if new_tree:
                    output = tree.nodes.new('NodeGroupOutput')
                    output.location = (node.location.x+340, node.location.y)
                    tree.links.new(node.outputs['Color'], output.inputs['Image'])
                    tree.nodes.active = node
                context.window_manager.pg_diagram_enabled = not self.stack
                diagram.ensure_routers()
                redraw(context)
                return {'FINISHED'}
            except Exception as exc:
                outcomes.failure(outcomes.target(context), self.bl_idname, exc)
                try:
                    if created_tree is not None:
                        context.scene.compositing_node_group = None
                        remove_node_group(created_tree)
                    elif tree is not None and existing_nodes is not None:
                        for item in tuple(tree.nodes):
                            if item.as_pointer() not in existing_nodes:
                                remove_node(tree, item)
                except Exception as cleanup_error:
                    # Keep Blender's normal undo snapshot if structural effects
                    # remain after failed compensation; cancellation would omit it.
                    outcomes.failure(outcomes.target(context), self.bl_idname + '.cleanup', cleanup_error)
                    return {'FINISHED'}
                return {'CANCELLED'}


    class PG_OT_add(AddGradeOperator, bpy.types.Operator):
        bl_idname = 'perceptual_grade.add_linear'
        bl_label = native('product.name')


    class PG_OT_add_stack(AddGradeOperator, bpy.types.Operator):
        bl_idname='perceptual_grade.add_stack'
        bl_label=native('stack.name')
        bl_description=native('tooltip.stack')
        stack=True


    class PG_OT_reset(bpy.types.Operator):
        bl_idname = 'perceptual_grade.reset_linear'
        bl_label = native('action.reset_linear.label')
        bl_translation_context = CONTEXT
        bl_description = native('tooltip.action.reset_linear')
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            node = nodes.active_grade(context)
            return node is not None and nodes.editable(node)

        def execute(self, context):
            try:
                nodes.reset_linear(nodes.active_grade(context))
                redraw(context)
                return {'FINISHED'}
            except ValueError as exc:
                outcomes.failure(outcomes.target(context), self.bl_idname, exc)
                return {'CANCELLED'}


    class PG_OT_reset_white(bpy.types.Operator):
        bl_idname = 'perceptual_grade.reset_white'
        bl_label = native('action.match_white.label')
        bl_translation_context = CONTEXT
        bl_description = native('tooltip.action.match_white')
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return PG_OT_reset.poll(context)

        def execute(self, context):
            try:
                wp = cm.white(nodes.grading_space(nodes.active_grade(context)))
                nodes.write_values(nodes.active_grade(context), dict(source_x=wp[0], source_y=wp[1]))
                redraw(context)
                return {'FINISHED'}
            except ValueError as exc:
                outcomes.failure(outcomes.target(context), self.bl_idname, exc)
                return {'CANCELLED'}


    class PG_OT_reset_directions(bpy.types.Operator):
        bl_idname='perceptual_grade.reset_directions'
        bl_label=native('action.reset_directions.label')
        bl_translation_context = CONTEXT
        bl_description=native('tooltip.action.reset_directions')
        bl_options={'REGISTER','UNDO'}

        @classmethod
        def poll(cls,context):return PG_OT_reset.poll(context)

        def execute(self,context):
            nodes.write_values(nodes.active_grade(context),
                {axis+'_'+control:0. for axis in ('cr','yb') for control in ('balance','spread','heading')})
            redraw(context);return {'FINISHED'}


    def diagram_target(context):
        """Use the clicked Grade even when another node is currently active."""
        clicked = getattr(context, 'pg_node', None)
        if nodes.is_grade(clicked):
            space = context.space_data
            return clicked if (space and space.type == 'NODE_EDITOR' and
                               space.edit_tree == clicked.id_data) else None
        return nodes.active_grade(context)


    class PG_OT_toggle_diagram(bpy.types.Operator):
        bl_idname = 'perceptual_grade.toggle_diagram'
        bl_label = native('action.toggle_panel.label')
        bl_translation_context = CONTEXT
        bl_description = native('tooltip.action.show_panel')

        @classmethod
        def description(cls, context, properties):
            editing = _application.component('graph.editing')
            state = editing.shown(context, diagram_target(context))
            return tip('tooltip.action.hide_panel' if state else 'tooltip.action.show_panel')

        @classmethod
        def poll(cls, context):
            return diagram_target(context) is not None

        def execute(self, context):
            editing = _application.component('graph.editing')
            target = diagram_target(context)
            if target is None:return {'CANCELLED'}
            show = not editing.shown(context, target)
            if nodes.is_grade(target):
                target.id_data.nodes.active = target
                target.select = True
                editing.close(context)
            wm = context.window_manager
            wm.pg_diagram_enabled = show
            if not wm.pg_diagram_enabled:
                editing.close(context)
                diagram.cleanup_hover_cursors()
            elif wm.pg_stage=='CHROMATIC':land_on_lightness(context)
            diagram.ensure_routers()
            redraw(context)
            return {'FINISHED'}


    class PG_OT_set_peak(bpy.types.Operator):
        bl_idname='perceptual_grade.set_peak'
        bl_label=native('action.set_peak.label')
        bl_translation_context = CONTEXT
        bl_options={'REGISTER','UNDO'}
        value: IntProperty(translation_context=CONTEXT, default=1000)

        @classmethod
        def description(cls, context, properties):
            return tip('tooltip.action.set_peak', value=properties.value)

        @classmethod
        def poll(cls,context):return PG_OT_reset.poll(context)

        def execute(self,context):
            node=nodes.active_grade(context)
            if self.value not in tm.PEAKS or self.value<2*node.pg_tone.reference_nits:return {'CANCELLED'}
            nodes.write_tone(node,dict(peak_nits=self.value))
            if context.scene.tool_settings.use_keyframe_insert_auto:nodes.keyframe_tone(node,'peak_nits')
            redraw(context);return {'FINISHED'}


    class PG_MT_peak(bpy.types.Menu):
        bl_idname='PG_MT_peak'
        bl_label=native('control.peak_nits.label')
        bl_translation_context = CONTEXT

        def draw(self,context):
            node=nodes.active_grade(context)
            if node is None:return
            for value in tm.PEAKS:
                row=self.layout.row();row.enabled=value>=2*node.pg_tone.reference_nits
                op=row.operator('perceptual_grade.set_peak',text=tr('unit.nits.format',value=value),icon='RADIOBUT_ON' if value==int(node.pg_tone.peak_nits) else 'RADIOBUT_OFF')
                op.value=value


    class PG_OT_reset_tone(bpy.types.Operator):
        bl_idname='perceptual_grade.reset_tone'
        bl_label=native('action.reset_tone.label')
        bl_translation_context = CONTEXT
        bl_description=native('tooltip.action.reset_tone')
        bl_options={'REGISTER','UNDO'}

        @classmethod
        def poll(cls,context):return PG_OT_reset.poll(context)

        def execute(self,context):
            node=nodes.active_grade(context);m=tm.model(nodes.read_tone(node),nodes.grading_space(node))
            values=dict(zip(tm.LEVEL_FIELDS,m.target_nits));values['fit']=0
            nodes.write_tone(node,values);redraw(context);return {'FINISHED'}


    class PG_OT_reset_chromatic(bpy.types.Operator):
        bl_idname='perceptual_grade.reset_chromatic'
        bl_label=native('action.reset_color.label')
        bl_translation_context = CONTEXT
        bl_description=native('tooltip.action.reset_color')
        bl_options={'REGISTER','UNDO'}

        @classmethod
        def poll(cls,context):return PG_OT_reset.poll(context)

        def execute(self,context):
            defaults=ch.Parameters();node=nodes.active_grade(context)
            nodes.write_chromatic(node,{n:getattr(defaults,n) for n in color_fields(node.pg_chromatic.family)})
            redraw(context);return {'FINISHED'}


    class PG_OT_reset_stage(bpy.types.Operator):
        bl_idname='perceptual_grade.reset_stage'
        bl_label=native('action.reset_stage.label')
        bl_translation_context = CONTEXT
        bl_description=native('tooltip.action.reset_stage')
        bl_options={'REGISTER','UNDO'}

        @classmethod
        def poll(cls,context):return PG_OT_reset.poll(context)

        def execute(self,context):
            node=nodes.active_grade(context);stage=context.window_manager.pg_stage
            if stage=='LINEAR':
                nodes.reset_linear(node)
            elif stage=='REMAP':
                defaults=tm.Parameters()
                nodes.write_tone(node,{n:getattr(defaults,n) for n in tm.PARAM_NAMES})
            else:
                defaults=ch.Parameters()
                nodes.write_chromatic(node,{n:getattr(defaults,n) for n in ch.PARAM_NAMES})
            redraw(context);return {'FINISHED'}


    def land_on_lightness(context):
        """A saved inspector page must not replace the Stage 3 landing page."""
        node=nodes.active_grade(context)
        if node is not None and nodes.editable(node):node.pg_chromatic.family='J'


    def stage_changed(_wm,context):
        if _wm.pg_stage=='CHROMATIC':land_on_lightness(context)
        # The router is maintained by the registration timer and diagram toggle.
        # Do not invoke a modal operator from an RNA property-update callback.
        redraw(context)


    class PG_OT_retry_recovery(bpy.types.Operator):
        bl_idname='perceptual_grade.retry_recovery'
        bl_label=native('recovery.retry')
        bl_description=native('recovery.retry.tooltip')
        bl_translation_context=CONTEXT

        def execute(self,context):
            return {'FINISHED'} if nodes.retry_recovery() else {'CANCELLED'}


    class PG_PT_main(bpy.types.Panel):
        bl_idname = 'PG_PT_main'
        bl_label = native('product.name')
        bl_translation_context = CONTEXT
        bl_space_type = 'NODE_EDITOR'
        bl_region_type = 'UI'
        bl_category = native('product.name')

        @classmethod
        def poll(cls, context):
            return supported_editor(context)

        def draw(self, context):
            layout = self.layout
            outcomes.draw(layout)
            current = outcomes.target(context)
            if current is not None:outcomes.draw(layout, current)
            if nodes.recovery_pending():
                status=nodes.recovery_status()
                if status['state']=='failed':
                    layout.label(text=tr('recovery.failed'),icon='ERROR')
                    layout.operator(PG_OT_retry_recovery.bl_idname,icon='FILE_REFRESH')
                else:
                    message=(tr('recovery.progress',completed=status['completed'],total=status['total'])
                             if status['total'] else tr('recovery.status'))
                    layout.label(text=message,icon='TIME')
                return
            tree = context.space_data.edit_tree
            candidate = tree.nodes.active if tree else None
            if candidate is not None and nodes.grades.draw_resource_action(layout, candidate):return
            stacks = _application.component('graph.stacks')
            stack=stacks.active(context)
            if stack is not None:
                stacks.draw(layout,stack,context)
                if nodes.active_grade(context) is None:return
            node = nodes.active_grade(context)
            try:
                space = nodes.working_space()
            except ValueError as exc:
                layout.label(text=tr('error.working_space.label'), icon='ERROR')
                layout.label(text=tr('outcome.working_space'))
                return
            if node is None:
                layout.operator('perceptual_grade.add_linear', icon='ADD')
                layout.operator('perceptual_grade.add_stack', icon='ADD')
                return
            if stack is None:
                nodes.grades.draw_selector(layout, node)
                nodes.grades.draw_evaluation(layout,node)
            else:
                layout.separator()
                layout.label(text=tr('stack.editing',grade=node.pg_grade.name,
                    direction=tr('direction.'+node.entry.direction.lower())),icon='GREASEPENCIL')
                layout.label(text=tr('scope.stack.output'))
            layout.prop(context.window_manager,'pg_stage',expand=True)
            row = layout.row(align=True)
            row.operator('perceptual_grade.toggle_diagram', text=tr('action.hide_panel.label' if context.window_manager.pg_diagram_enabled else 'action.show_panel.label'), icon='HIDE_OFF')
            row.operator('perceptual_grade.reset_stage', text=tr('action.reset_stage.label'), icon='LOOP_BACK')
            if context.window_manager.pg_diagram_enabled:
                layout.prop(context.window_manager,{'LINEAR':'pg_diagram_size','REMAP':'pg_tone_width','CHROMATIC':'pg_color_width'}[context.window_manager.pg_stage],text=tr('diagram.width.label'))


    class PG_PT_linear(bpy.types.Panel):
        bl_idname = 'PG_PT_linear'
        bl_label = native('stage.linear.label')
        bl_translation_context = CONTEXT
        bl_space_type = 'NODE_EDITOR'
        bl_region_type = 'UI'
        bl_category = native('product.name')
        bl_parent_id = 'PG_PT_main'

        @classmethod
        def poll(cls, context):
            node = nodes.active_grade(context) if supported_editor(context) else None
            return node is not None and nodes.resource_action(node) is None

        def draw(self, context):
            node = nodes.active_grade(context);p = nodes.grades.data_owner(node).pg_linear;layout = self.layout
            layout.enabled = nodes.editable(node) and not bool(node.get('_pg_error'))
            layout.prop(p, 'exposure', slider=True)
            layout.separator()
            row = layout.row(align=True)
            row.prop(p, 'source_x', text=tr('control.source_x.label'));row.prop(p, 'source_y', text=tr('control.source_y.label'))
            layout.operator('perceptual_grade.reset_white', text=tr('action.match_white.label'))
            row=layout.row();row.enabled=False
            row.prop(nodes.grades.data_owner(node), 'pg_working_white')
            layout.label(text=tr('context.d65_only'))
            layout.separator()
            layout.prop(p, 'cr_balance_display', slider=True)
            layout.prop(p, 'yb_balance_display', slider=True)
            layout.prop(p, 'cr_spread',slider=True);layout.prop(p, 'yb_spread',slider=True)
            layout.prop(p, 'cr_heading',slider=True);layout.prop(p, 'yb_heading',slider=True)
            layout.operator('perceptual_grade.reset_directions',icon='LOOP_BACK')


    class PG_PT_remap(bpy.types.Panel):
        bl_idname = 'PG_PT_remap'
        bl_label = native('stage.tone.label')
        bl_translation_context = CONTEXT
        bl_space_type = 'NODE_EDITOR'
        bl_region_type = 'UI'
        bl_category = native('product.name')
        bl_parent_id = 'PG_PT_main'

        @classmethod
        def poll(cls, context):
            return PG_PT_linear.poll(context)

        def draw(self, context):
            node=nodes.active_grade(context);p=nodes.grades.data_owner(node).pg_tone;layout=self.layout
            layout.enabled=nodes.editable(node) and not bool(node.get('_pg_error'))
            # This is the very same RNA property drawn in Stage 1, never a proxy.
            layout.prop(nodes.grades.data_owner(node).pg_linear,'exposure',slider=True)
            layout.separator()
            for name in tm.LEVEL_FIELDS:layout.prop(p,name,slider=True)
            layout.prop(p,'fit',text=tr('control.remap_fit.label'),slider=True)
            layout.operator('perceptual_grade.reset_tone',icon='LOOP_BACK')
            layout.separator()
            row=layout.row();row.enabled=False;row.prop(p,'reference_nits')
            layout.label(text=tr('context.d65_only'))
            layout.menu('PG_MT_peak',text=tr('peak.menu.label',value=int(p.peak_nits)))


    def color_fields(family):
        """Every color coefficient is editable numerically, in DCTL tone order."""
        if family in ('J','C'):
            stem=family.lower()
            fields=[f'{stem}_{driver}_{tone}' for tone in range(4) for driver in ('low','high','a','b')]
            if family=='C':fields.append('global_gain')
            return fields+[stem+'_fit']
        return [name for name in ch.PARAM_NAMES if name.startswith(('adaptance_','web_'))]



    class PG_PT_chromatic(bpy.types.Panel):
        bl_idname = 'PG_PT_chromatic'
        bl_label = native('stage.color.label')
        bl_translation_context = CONTEXT
        bl_space_type = 'NODE_EDITOR'
        bl_region_type = 'UI'
        bl_category = native('product.name')
        bl_parent_id = 'PG_PT_main'
        bl_options = {'DEFAULT_CLOSED'}

        @classmethod
        def poll(cls, context):
            return PG_PT_linear.poll(context)

        def draw(self, context):
            node=nodes.active_grade(context);layout=self.layout;ui=node.pg_chromatic
            layout.enabled=nodes.editable(node) and not bool(node.get('_pg_error'))
            layout.prop(ui,'family',text='')
            values=nodes.grades.data_owner(node).pg_chromatic
            if ui.family=='WARP':
                layout.prop(values,'sidebar_adaptance_half_saturation',slider=True)
                for tone,name in enumerate(ch.TONES):
                    group=layout.box()
                    group.label(text=tr('tone.'+name.lower()))
                    row=group.row(align=True)
                    row.label(text=tr('warp.center'))
                    for axis in 'ab':row.prop(values,f'sidebar_adaptance_{axis}_{tone}',text=axis,slider=True)
                    for handle,label in enumerate('RYGCBM'):
                        row=group.row(align=True)
                        row.label(text=label)
                        for family,message in (('chroma','diagram.warp.chroma_stops'),('hue','warp.hue'),('scale','warp.scale')):
                            row.prop(values,f'sidebar_web_{family}_{tone}_{handle}',text=tr(message),slider=True)
                layout.operator('perceptual_grade.reset_chromatic',icon='LOOP_BACK')
                return
            fields=color_fields(ui.family);i=0
            while i<len(fields):
                field=fields[i]
                if '_a_' in field and i+1<len(fields) and fields[i+1]==field.replace('_a_','_b_'):
                    row=layout.row(align=True)
                    row.prop(values,field,slider=True);row.prop(values,fields[i+1],slider=True);i+=2
                else:layout.prop(values,field,slider=True);i+=1
            layout.operator('perceptual_grade.reset_chromatic',icon='LOOP_BACK')


    CLASSES = (PG_OT_add, PG_OT_add_stack, PG_OT_reset, PG_OT_reset_white, PG_OT_reset_directions, PG_OT_toggle_diagram,
               PG_OT_set_peak,PG_MT_peak,PG_OT_reset_tone,PG_OT_reset_chromatic,PG_OT_reset_stage,
               PG_OT_retry_recovery, PG_PT_main, PG_PT_linear, PG_PT_remap, PG_PT_chromatic)


    MENU_TYPES=(bpy.types.NODE_MT_category_shader_color,bpy.types.NODE_MT_category_compositor_color)


    def menu(self, context):
        if supported_editor(context):
            self.layout.separator()
            self.layout.operator('perceptual_grade.add_linear', icon=nodes.grades.PGGradeData.bl_icon)
            self.layout.operator('perceptual_grade.add_stack', icon=nodes.grades.PGGradeData.bl_icon)


    menu = outcomes.callback(menu, control='menu')

    def register():
        nonlocal _menu_registered, _diagram_registered
        wm = bpy.types.WindowManager
        wm.pg_diagram_enabled = BoolProperty(translation_context=CONTEXT, default=False, options={'SKIP_SAVE'})
        wm.pg_diagram_size = IntProperty(translation_context=CONTEXT, name=native('diagram.width.label'), default=300, min=300, max=900, options={'SKIP_SAVE'})
        wm.pg_diagram_x = FloatProperty(translation_context=CONTEXT, default=24, options={'SKIP_SAVE'})
        wm.pg_diagram_y = FloatProperty(translation_context=CONTEXT, default=24, options={'SKIP_SAVE'})
        wm.pg_stage=outcomes.property(EnumProperty(translation_context=CONTEXT, name=native('stage.label'),items=[('LINEAR',native('stage.linear.label'),native('tooltip.stage.linear')),('REMAP',native('stage.tone.label'),native('tooltip.stage.tone')),('CHROMATIC',native('stage.color.label'),native('tooltip.stage.color'))],default='LINEAR',options={'SKIP_SAVE'},update=stage_changed), control='stage')
        wm.pg_tone_width=IntProperty(translation_context=CONTEXT, name=native('diagram.tone_width.label'),default=390,min=360,max=1200,options={'SKIP_SAVE'})
        wm.pg_color_width=IntProperty(translation_context=CONTEXT, name=native('diagram.color_width.label'),default=390,min=360,max=1400,options={'SKIP_SAVE'})
        wm.pg_selected_tone=IntProperty(translation_context=CONTEXT, default=3,min=0,max=4,options={'SKIP_SAVE'})
        try:
            for cls in CLASSES:
                _application.register_class(cls)
                _classes.append(cls)
            diagram.register()
            _diagram_registered = True
            for menu_type in MENU_TYPES:menu_type.append(menu)
            _menu_registered = True
            # A window-modal event router precedes node UI buttons and keymaps.
        except Exception:
            unregister()
            raise


    def unregister():
        nonlocal _menu_registered, _diagram_registered
        _remove_keymap_items = _application.component('integration.lifecycle')._remove_keymap_items
        _remove_keymap_items(_keymaps)
        if _menu_registered:
            for menu_type in MENU_TYPES:menu_type.remove(menu)
            _menu_registered = False
        if _diagram_registered:
            diagram.unregister()
            _diagram_registered = False
        for cls in reversed(_classes):
            _application.unregister_class(cls)
        _classes.clear()
        scopes.clear()
        comparison = _application.component('scopes.reference')
        comparison.clear()
        for name in ('pg_diagram_enabled', 'pg_diagram_size', 'pg_diagram_x', 'pg_diagram_y', 'pg_stage','pg_tone_width','pg_selected_tone','pg_color_width'):
            if hasattr(bpy.types.WindowManager, name):
                delattr(bpy.types.WindowManager, name)

    def _set_export_CLASSES(value):
        nonlocal CLASSES
        CLASSES = value

    def _set_export_MENU_TYPES(value):
        nonlocal MENU_TYPES
        MENU_TYPES = value

    def _set_export_RELEASE(value):
        nonlocal RELEASE
        RELEASE = value

    def _set_export__classes(value):
        nonlocal _classes
        _classes = value

    def _set_export__diagram_registered(value):
        nonlocal _diagram_registered
        _diagram_registered = value

    def _set_export__keymaps(value):
        nonlocal _keymaps
        _keymaps = value

    def _set_export__menu_registered(value):
        nonlocal _menu_registered
        _menu_registered = value

    def _set_export_prepare_linear_properties(value):
        nonlocal prepare_linear_properties
        prepare_linear_properties = value

    def _set_export__warp_source_angles(value):
        nonlocal _warp_source_angles
        _warp_source_angles = value

    def _set_export_warp_hue_origin(value):
        nonlocal warp_hue_origin
        warp_hue_origin = value

    def _set_export_warp_hue_value(value):
        nonlocal warp_hue_value
        warp_hue_value = value

    def _set_export_warp_hue_delta(value):
        nonlocal warp_hue_delta
        warp_hue_delta = value

    def _set_export_warp_sidebar_property(value):
        nonlocal warp_sidebar_property
        warp_sidebar_property = value

    def _set_export_prepare_warp_properties(value):
        nonlocal prepare_warp_properties
        prepare_warp_properties = value

    def _set_export_supported_editor(value):
        nonlocal supported_editor
        supported_editor = value

    def _set_export_redraw(value):
        nonlocal redraw
        redraw = value

    def _set_export_AddGradeOperator(value):
        nonlocal AddGradeOperator
        AddGradeOperator = value

    def _set_export_PG_OT_add(value):
        nonlocal PG_OT_add
        PG_OT_add = value

    def _set_export_PG_OT_add_stack(value):
        nonlocal PG_OT_add_stack
        PG_OT_add_stack = value

    def _set_export_PG_OT_reset(value):
        nonlocal PG_OT_reset
        PG_OT_reset = value

    def _set_export_PG_OT_reset_white(value):
        nonlocal PG_OT_reset_white
        PG_OT_reset_white = value

    def _set_export_PG_OT_reset_directions(value):
        nonlocal PG_OT_reset_directions
        PG_OT_reset_directions = value

    def _set_export_diagram_target(value):
        nonlocal diagram_target
        diagram_target = value

    def _set_export_PG_OT_toggle_diagram(value):
        nonlocal PG_OT_toggle_diagram
        PG_OT_toggle_diagram = value

    def _set_export_PG_OT_set_peak(value):
        nonlocal PG_OT_set_peak
        PG_OT_set_peak = value

    def _set_export_PG_MT_peak(value):
        nonlocal PG_MT_peak
        PG_MT_peak = value

    def _set_export_PG_OT_reset_tone(value):
        nonlocal PG_OT_reset_tone
        PG_OT_reset_tone = value

    def _set_export_PG_OT_reset_chromatic(value):
        nonlocal PG_OT_reset_chromatic
        PG_OT_reset_chromatic = value

    def _set_export_PG_OT_reset_stage(value):
        nonlocal PG_OT_reset_stage
        PG_OT_reset_stage = value

    def _set_export_land_on_lightness(value):
        nonlocal land_on_lightness
        land_on_lightness = value

    def _set_export_stage_changed(value):
        nonlocal stage_changed
        stage_changed = value

    def _set_export_PG_PT_main(value):
        nonlocal PG_PT_main
        PG_PT_main = value

    def _set_export_PG_PT_linear(value):
        nonlocal PG_PT_linear
        PG_PT_linear = value

    def _set_export_PG_PT_remap(value):
        nonlocal PG_PT_remap
        PG_PT_remap = value

    def _set_export_color_fields(value):
        nonlocal color_fields
        color_fields = value

    def _set_export_PG_PT_chromatic(value):
        nonlocal PG_PT_chromatic
        PG_PT_chromatic = value

    def _set_export_menu(value):
        nonlocal menu
        menu = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component(readers={
        "__name__": lambda: __name__,
        'AddGradeOperator': lambda: AddGradeOperator,
        'BoolProperty': lambda: BoolProperty,
        'CLASSES': lambda: CLASSES,
        'CONTEXT': lambda: CONTEXT,
        'EnumProperty': lambda: EnumProperty,
        'FloatProperty': lambda: FloatProperty,
        'IntProperty': lambda: IntProperty,
        'MENU_TYPES': lambda: MENU_TYPES,
        'PG_MT_peak': lambda: PG_MT_peak,
        'PG_OT_add': lambda: PG_OT_add,
        'PG_OT_add_stack': lambda: PG_OT_add_stack,
        'PG_OT_reset': lambda: PG_OT_reset,
        'PG_OT_reset_chromatic': lambda: PG_OT_reset_chromatic,
        'PG_OT_reset_directions': lambda: PG_OT_reset_directions,
        'PG_OT_reset_stage': lambda: PG_OT_reset_stage,
        'PG_OT_reset_tone': lambda: PG_OT_reset_tone,
        'PG_OT_reset_white': lambda: PG_OT_reset_white,
        'PG_OT_set_peak': lambda: PG_OT_set_peak,
        'PG_OT_toggle_diagram': lambda: PG_OT_toggle_diagram,
        'PG_PT_chromatic': lambda: PG_PT_chromatic,
        'PG_PT_linear': lambda: PG_PT_linear,
        'PG_PT_main': lambda: PG_PT_main,
        'PG_PT_remap': lambda: PG_PT_remap,
        'RELEASE': lambda: RELEASE,
        '_classes': lambda: _classes,
        '_diagram_registered': lambda: _diagram_registered,
        '_keymaps': lambda: _keymaps,
        '_menu_registered': lambda: _menu_registered,
        '_warp_source_angles': lambda: _warp_source_angles,
        'bpy': lambda: bpy,
        'ch': lambda: ch,
        'cm': lambda: cm,
        'color_fields': lambda: color_fields,
        'controls': lambda: controls,
        'diagnostic_text': lambda: diagnostic_text,
        'diagram': lambda: diagram,
        'diagram_target': lambda: diagram_target,
        'error_text': lambda: error_text,
        'land_on_lightness': lambda: land_on_lightness,
        'lru_cache': lambda: lru_cache,
        'math': lambda: math,
        'menu': lambda: menu,
        'native': lambda: native,
        'nodes': lambda: nodes,
        'np': lambda: np,
        'prepare_linear_properties': lambda: prepare_linear_properties,
        'prepare_warp_properties': lambda: prepare_warp_properties,
        'redraw': lambda: redraw,
        'register': lambda: register,
        'remove_node': lambda: remove_node,
        'remove_node_group': lambda: remove_node_group,
        'scopes': lambda: scopes,
        'stage_changed': lambda: stage_changed,
        'supported_editor': lambda: supported_editor,
        'tip': lambda: tip,
        'tm': lambda: tm,
        'tr': lambda: tr,
        'unregister': lambda: unregister,
        'warp_hue_delta': lambda: warp_hue_delta,
        'warp_hue_origin': lambda: warp_hue_origin,
        'warp_hue_value': lambda: warp_hue_value,
        'warp_sidebar_property': lambda: warp_sidebar_property,
    }, writers={
        'CLASSES': _set_export_CLASSES,
        'MENU_TYPES': _set_export_MENU_TYPES,
        'RELEASE': _set_export_RELEASE,
        '_classes': _set_export__classes,
        '_diagram_registered': _set_export__diagram_registered,
        '_keymaps': _set_export__keymaps,
        '_menu_registered': _set_export__menu_registered,
    'prepare_linear_properties': _set_export_prepare_linear_properties, '_warp_source_angles': _set_export__warp_source_angles, 'warp_hue_origin': _set_export_warp_hue_origin, 'warp_hue_value': _set_export_warp_hue_value, 'warp_hue_delta': _set_export_warp_hue_delta, 'warp_sidebar_property': _set_export_warp_sidebar_property, 'prepare_warp_properties': _set_export_prepare_warp_properties, 'supported_editor': _set_export_supported_editor, 'redraw': _set_export_redraw, 'AddGradeOperator': _set_export_AddGradeOperator, 'PG_OT_add': _set_export_PG_OT_add, 'PG_OT_add_stack': _set_export_PG_OT_add_stack, 'PG_OT_reset': _set_export_PG_OT_reset, 'PG_OT_reset_white': _set_export_PG_OT_reset_white, 'PG_OT_reset_directions': _set_export_PG_OT_reset_directions, 'diagram_target': _set_export_diagram_target, 'PG_OT_toggle_diagram': _set_export_PG_OT_toggle_diagram, 'PG_OT_set_peak': _set_export_PG_OT_set_peak, 'PG_MT_peak': _set_export_PG_MT_peak, 'PG_OT_reset_tone': _set_export_PG_OT_reset_tone, 'PG_OT_reset_chromatic': _set_export_PG_OT_reset_chromatic, 'PG_OT_reset_stage': _set_export_PG_OT_reset_stage, 'land_on_lightness': _set_export_land_on_lightness, 'stage_changed': _set_export_stage_changed, 'PG_PT_main': _set_export_PG_PT_main, 'PG_PT_linear': _set_export_PG_PT_linear, 'PG_PT_remap': _set_export_PG_PT_remap, 'color_fields': _set_export_color_fields, 'PG_PT_chromatic': _set_export_PG_PT_chromatic, 'menu': _set_export_menu, 'register': _set_export_register, 'unregister': _set_export_unregister})
