"""Node-local ordered Grade references; never a shared Stack datablock."""

from ..integration.components import Component

def create(_application):
    import bpy
    from uuid import uuid4
    from bpy.props import CollectionProperty, PointerProperty, EnumProperty, IntProperty, BoolProperty, StringProperty
    native = _application.component('localization.i18n').native; tr = _application.component('localization.i18n').tr; tip = _application.component('localization.i18n').tip; CONTEXT = _application.component('localization.i18n').CONTEXT
    from ...perceptual_grade_core.model.transforms import Stack
    grades = _application.component('graph.grades'); editing = _application.component('graph.editing')

    TAG='_pg_stack'
    SCHEMA=8
    KINDS={'ShaderNodeTree':'PGShaderStack','CompositorNodeTree':'PGCompositorStack'}
    _classes=[]


    def core():
        nodes = _application.component('graph.nodes')
        return nodes


    def is_stack(node):
        return bool(node and node.get(TAG)==SCHEMA and getattr(node,'node_tree',None)
                    and node.node_tree.get('_pg_transform_kind')=='STACK'
                    and node.node_tree.get('_pg_cube_schema') == core().cube_nodes.SCHEMA
                    and _application.component('graph.output_resources').is_current(node))


    def active(context):
        space=context.space_data
        tree=space.edit_tree if space and space.type=='NODE_EDITOR' else None
        node=tree.nodes.active if tree else None
        return node if is_stack(node) else None


    def entries(node):
        return [(item.grade,item.enabled,item.direction,item.key,
                 tuple(getattr(item.focus,name) for name in grades.FOCUS_FIELDS))
                for item in node.pg_stack_entries]


    def _write_entries(node,values):
        with grades.guarded(node):
            node.pg_stack_entries.clear()
            for grade,enabled,direction,key,focus in values:
                item=node.pg_stack_entries.add();item.grade=grade
                item.enabled=enabled;item.direction=direction;item.key=key
                for name,value in zip(grades.FOCUS_FIELDS,focus):setattr(item.focus,name,value)


    def assign(node,values):
        transactions = _application.component('ownership.transactions')
        if node.id_data.is_evaluated or not grades.local(node.id_data):
            raise ValueError('Cannot replace borrowed or linked Stack references')
        before = entries(node)
        values = list(values)
        record = {'node': transactions.node_key(node), 'values': transactions.stack_values(before),
                  'after_values': transactions.stack_values(values)}
        tx = transactions.Transaction('stack', record)
        try:
            tx.apply('stack_replace_by_occurrence', lambda: _write_entries(node, values),
                     lambda: _write_entries(node, before),
                     before_edges=tuple(transactions.id_key(value[0]) for value in before if value[0] is not None),
                     after_edges=tuple(transactions.id_key(value[0]) for value in values if value[0] is not None))
            tx.commit()
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()


    def repair_assignment(record):
        transactions = _application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        if tree is None: return
        node = transactions.find_node(tree, record['node'])
        if node is None: raise RuntimeError('Stack repair lost its owning node')
        _write_entries(node, transactions.restore_stack_values(record['values']))
        verify_assignment(record)


    def verify_assignment(record, after=False):
        transactions = _application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        if tree is None: return
        node = transactions.find_node(tree, record['node'])
        if node is None: raise RuntimeError('Stack receipt lost its owning node')
        values = record['after_values'] if after else record['values']
        if entries(node) != transactions.restore_stack_values(values):
            raise RuntimeError('Stack occurrence references do not match their receipt')


    def model(node):
        n=core()
        items=[item for item in node.pg_stack_entries if item.enabled and grades.is_data(item.grade)]
        return Stack([n.settings(item.grade) for item in items],'Linear Rec.709',[item.direction for item in items])


    def bind_dependencies(node):
        """A scene dependency for each referenced Grade's animation."""
        path='["_pg_stack_dependency"]'
        node['_pg_stack_dependency']=0.
        curve=node.driver_add(path);driver=curve.driver;driver.type='SUM'
        for variable in list(driver.variables):driver.variables.remove(variable)
        selected=[]
        for item in node.pg_stack_entries:
            if item.enabled and grades.is_data(item.grade) and item.grade not in selected:selected.append(item.grade)
        for index,grade in enumerate(selected):
            variable=driver.variables.new();variable.name='grade_'+str(index);variable.type='SINGLE_PROP'
            target=variable.targets[0];target.id_type='NODETREE';target.id=grade
            target.data_path='["'+grades.MATRIX+'"][0]'


    def changed(item,context):
        if core().recovery_pending() or grades._converting:return
        for node in item.id_data.nodes:
            if not is_stack(node) or node.as_pointer() in grades._busy:continue
            if any(entry.as_pointer()==item.as_pointer() for entry in node.pg_stack_entries):
                if not item.key:item.key=uuid4().hex
                if item.grade is not None and not grades.is_data(item.grade):
                    with grades.guarded(node):item.grade=None
                editing.prune()
                bind_dependencies(node);refresh(node,True);grades.redraw()
                break


    def publish(grade):
        if core().recovery_pending():return
        for node in core().all_nodes():
            if is_stack(node) and any(item.enabled and item.grade==grade for item in node.pg_stack_entries):
                refresh(node)


    def refresh(node,force=False,immediate=False):
        n=core()
        if n.recovery_pending() or not is_stack(node) or not grades.local(node.id_data):return
        # Advance the Stack's own semantic revision before any deferred output
        # work. Bake requests observe this source locally; they are never found
        # or cancelled from an upstream Grade callback.
        n.observe_intent(node)
        if not n.resources_ready(node):return
        if not n.cube_nodes.request_intent(node,n.output_intent(node)):return
        editing.prune()
        presentation = _application.component('graph.presentation'); compositor = _application.component('graph.compositor')
        presentation.node_options(node)
        update_node_title = _application.component('localization.i18n').update_node_title
        update_node_title(node)
        scene_space=n.working_space()
        try:
            evaluator=model(node)
        except (ValueError,ArithmeticError) as exc:
            if node.get('_pg_error','')!=str(exc):node['_pg_error']=str(exc)
            n.cube_nodes.stop(n.cube_nodes._states.pop(node.as_pointer(),None))
            return
        if node.get('_pg_error',''):node['_pg_error']=''
        signature=(scene_space,evaluator.cache_key)
        n.cube_nodes.request(node,evaluator,force=n._rendering)
        if not immediate and not n._rendering and compositor.defer_update(node,force):return
        if not force and n._signatures.get(node.as_pointer())==signature:return
        signal_before=n.cube_nodes.signal_signature(node)
        # While a different core is displayed, its endpoint matrices remain part
        # of that same snapshot. Independent endpoint edits need no new LUT.
        if n.cube_nodes.installed_matches(node,evaluator):
            if not n.cube_nodes.commit_rows(node,evaluator):return
        n._signatures[node.as_pointer()]=signature
        n.cube_nodes.request(node,evaluator)
        if force or n.cube_nodes.signal_signature(node)!=signal_before:compositor.tag_output(node)


    def initialize(node):
        n=core();node.node_tree=n.cube_nodes.make_kernel(node.id_data.bl_idname)
        node[TAG]=SCHEMA;node.name=tr('node.stack.title');node.label=tr('node.stack.title')
        node['_pg_auto_label']=node.label;node.width=220;node.show_options=True
        node.use_custom_color=True;node.color=(.13,.28,.29)
        bind_dependencies(node)
        for key in ('_pg_pending_init','_pg_pending_copy'):
            if key in node:del node[key]
        refresh(node,True)


    def create(tree):
        if tree.bl_idname not in KINDS or tree.get(core().KERNEL):raise ValueError(tr('error.add_location'))
        node=tree.nodes.new(KINDS[tree.bl_idname]);initialize(node)
        for other in tree.nodes:other.select=other==node
        tree.nodes.active=node
        return node


    def draw(layout,node,context=None):
        context=context or bpy.context
        layout.enabled=grades.local(node.id_data)
        if grades.draw_resource_action(layout,node):return
        layout.context_pointer_set('pg_stack_node',node)
        for index,item in enumerate(node.pg_stack_entries):
            box=layout.box();row=box.row(align=True)
            row.label(text=str(index+1))
            row.template_ID(item,'grade')
            row=box.row(align=True)
            row.prop(item,'enabled',text='',icon='HIDE_OFF' if item.enabled else 'HIDE_ON',toggle=True)
            row.prop(item,'direction',text='')
            target=editing.active(context,node)
            pencil=row.row(align=True)
            pencil.enabled=item.enabled and grades.is_data(item.grade) and grades.local(item.grade)
            op=pencil.operator('perceptual_grade.stack_edit',text='',icon='GREASEPENCIL',
                depress=bool(target and target.key==item.key and editing.shown(context,target)))
            op.index=index
            for action,icon in (('UP','TRIA_UP'),('DOWN','TRIA_DOWN'),('REMOVE','X')):
                button=row.row(align=True)
                button.enabled=(action!='UP' or index>0) and (action!='DOWN' or index+1<len(node.pg_stack_entries))
                op=button.operator(ACTIONS[action].bl_idname,text='',icon=icon)
                op.index=index
        layout.operator('perceptual_grade.stack_add',text=tr('stack.add'),icon='ADD')
        _application.component('graph.bake').draw_action(layout, node, context)
        lut = _application.component('graph.lut')
        lut.draw_error(layout,node)


    def focus_changed(focus,context):
        if core().recovery_pending():return
        for node in getattr(focus.id_data,'nodes',()):
            if not is_stack(node):continue
            for item in node.pg_stack_entries:
                if item.focus.as_pointer()==focus.as_pointer() and grades.is_data(item.grade):
                    grades.redraw();return
        grades.redraw()


    class PGStackFocus(bpy.types.PropertyGroup):
        family: EnumProperty(name=native('family.label'),translation_context=CONTEXT,
            items=[('J',native('family.j.label'),''),('C',native('family.c.label'),''),
                   ('WARP',native('family.warp.label'),'')],
            default='J',update=focus_changed)
        active_handle: IntProperty(default=-1,min=-1,max=5,update=focus_changed)
        active_tone: IntProperty(default=1,min=0,max=3,update=focus_changed)
        active_chroma: EnumProperty(items=[('low','Cₗ',''),('high','Cₕ','')],default='low',
            update=focus_changed)


    class PGStackEntry(bpy.types.PropertyGroup):
        key: StringProperty(options={'HIDDEN'})
        enabled: BoolProperty(name=native('stack.enabled'),description=native('tooltip.stack.enabled'),
                              translation_context=CONTEXT,default=True,update=changed,options=set())
        direction: EnumProperty(name=native('direction.label'),translation_context=CONTEXT,
            items=[('FORWARD',native('direction.forward'),''),('INVERSE',native('direction.inverse'),'')],
            default='FORWARD',update=changed,options=set())
        focus: PointerProperty(type=PGStackFocus)
        grade: PointerProperty(type=bpy.types.NodeTree,name=native('grade.data.label'),
                               poll=lambda self,value:grades.is_data(value),update=changed)



    class StackNode:
        bl_label=native('node.stack.title')
        bl_description=native('tooltip.stack')
        bl_translation_context=CONTEXT

        def init(self,context):
            if not grades._converting:
                self[TAG]=SCHEMA;self['_pg_pending_init']=True

        def copy(self,original):
            core().copy_output_resources(self,original)

        def free(self):
            grades.release_node_references(self)

        def draw_buttons(self,context,layout):
            if core().recovery_pending():return
            draw(layout,self,context)

        def draw_buttons_ext(self,context,layout):
            if core().recovery_pending():return
            draw(layout,self,context)


    class PGShaderStack(StackNode,bpy.types.ShaderNodeCustomGroup):
        bl_idname='PGShaderStack'


    class PGCompositorStack(StackNode,bpy.types.CompositorNodeCustomGroup):
        bl_idname='PGCompositorStack'


    class StackAction:
        bl_translation_context=CONTEXT
        bl_options={'REGISTER','UNDO'}

        @classmethod
        def description(cls,context,properties):
            return tip('tooltip.stack.'+cls.action.lower())

        @classmethod
        def poll(cls,context):
            node=getattr(context,'pg_stack_node',None) or active(context)
            return is_stack(node) and grades.local(node.id_data)

        def execute(self,context):
            node=getattr(context,'pg_stack_node',None) or active(context)
            collection=node.pg_stack_entries
            with grades.guarded(node):
                if self.action=='ADD':collection.add().key=uuid4().hex
                elif 0<=self.index<len(collection):
                    if self.action=='REMOVE':collection.remove(self.index)
                    else:
                        target=self.index+(-1 if self.action=='UP' else 1)
                        if 0<=target<len(collection):collection.move(self.index,target)
            editing.prune()
            bind_dependencies(node);refresh(node,True);grades.redraw()
            return {'FINISHED'}


    class PG_OT_stack_add(StackAction,bpy.types.Operator):
        bl_idname='perceptual_grade.stack_add'
        bl_label=native('stack.add')
        action='ADD'


    class PG_OT_stack_remove(StackAction,bpy.types.Operator):
        bl_idname='perceptual_grade.stack_remove'
        bl_label=native('stack.remove')
        action='REMOVE'


    class PG_OT_stack_up(StackAction,bpy.types.Operator):
        bl_idname='perceptual_grade.stack_up'
        bl_label=native('stack.up')
        action='UP'


    class PG_OT_stack_down(StackAction,bpy.types.Operator):
        bl_idname='perceptual_grade.stack_down'
        bl_label=native('stack.down')
        action='DOWN'


    ACTIONS={cls.action:cls for cls in (PG_OT_stack_add,PG_OT_stack_remove,PG_OT_stack_up,PG_OT_stack_down)}
    for cls in ACTIONS.values():
        cls.__annotations__={'index':IntProperty(default=0,options={'SKIP_SAVE'})}


    class PG_OT_stack_edit(bpy.types.Operator):
        bl_idname='perceptual_grade.stack_edit'
        bl_label=native('stack.edit_grade')
        bl_description=native('tooltip.stack.edit_grade')
        bl_translation_context=CONTEXT
        bl_options={'INTERNAL'}
        index: IntProperty(default=0,options={'SKIP_SAVE'})

        @classmethod
        def poll(cls,context):
            return StackAction.poll(context) and editing.area_key(context) is not None

        def execute(self,context):
            node=getattr(context,'pg_stack_node',None) or active(context)
            if not 0<=self.index<len(node.pg_stack_entries):return {'CANCELLED'}
            item=node.pg_stack_entries[self.index]
            if not item.enabled or not grades.is_data(item.grade) or not grades.local(item.grade):
                return {'CANCELLED'}
            editing.toggle(context,node,item)
            diagram = _application.component('ui.diagram')
            diagram.ensure_routers()
            grades.redraw()
            return {'FINISHED'}


    def register():
        for cls in (PGStackFocus,PGStackEntry,PGShaderStack,PGCompositorStack,*ACTIONS.values(),PG_OT_stack_edit):
            _application.register_class(cls);_classes.append(cls)
        bpy.types.Node.pg_stack_entries=CollectionProperty(type=PGStackEntry)


    def unregister():
        editing.clear()
        if hasattr(bpy.types.Node,'pg_stack_entries'):del bpy.types.Node.pg_stack_entries
        for cls in reversed(_classes):_application.unregister_class(cls)
        _classes.clear()

    def _set_export_ACTIONS(value):
        nonlocal ACTIONS
        ACTIONS = value

    def _set_export_KINDS(value):
        nonlocal KINDS
        KINDS = value

    def _set_export_SCHEMA(value):
        nonlocal SCHEMA
        SCHEMA = value

    def _set_export_TAG(value):
        nonlocal TAG
        TAG = value

    def _set_export__classes(value):
        nonlocal _classes
        _classes = value

    def _set_export_core(value):
        nonlocal core
        core = value

    def _set_export_is_stack(value):
        nonlocal is_stack
        is_stack = value

    def _set_export_active(value):
        nonlocal active
        active = value

    def _set_export_entries(value):
        nonlocal entries
        entries = value

    def _set_export__write_entries(value):
        nonlocal _write_entries
        _write_entries = value

    def _set_export_assign(value):
        nonlocal assign
        assign = value

    def _set_export_repair_assignment(value):
        nonlocal repair_assignment
        repair_assignment = value

    def _set_export_verify_assignment(value):
        nonlocal verify_assignment
        verify_assignment = value

    def _set_export_model(value):
        nonlocal model
        model = value

    def _set_export_bind_dependencies(value):
        nonlocal bind_dependencies
        bind_dependencies = value

    def _set_export_changed(value):
        nonlocal changed
        changed = value

    def _set_export_publish(value):
        nonlocal publish
        publish = value

    def _set_export_refresh(value):
        nonlocal refresh
        refresh = value

    def _set_export_initialize(value):
        nonlocal initialize
        initialize = value

    def _set_export_create(value):
        nonlocal create
        create = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export_focus_changed(value):
        nonlocal focus_changed
        focus_changed = value

    def _set_export_PGStackFocus(value):
        nonlocal PGStackFocus
        PGStackFocus = value

    def _set_export_PGStackEntry(value):
        nonlocal PGStackEntry
        PGStackEntry = value

    def _set_export_StackNode(value):
        nonlocal StackNode
        StackNode = value

    def _set_export_PGShaderStack(value):
        nonlocal PGShaderStack
        PGShaderStack = value

    def _set_export_PGCompositorStack(value):
        nonlocal PGCompositorStack
        PGCompositorStack = value

    def _set_export_StackAction(value):
        nonlocal StackAction
        StackAction = value

    def _set_export_PG_OT_stack_add(value):
        nonlocal PG_OT_stack_add
        PG_OT_stack_add = value

    def _set_export_PG_OT_stack_remove(value):
        nonlocal PG_OT_stack_remove
        PG_OT_stack_remove = value

    def _set_export_PG_OT_stack_up(value):
        nonlocal PG_OT_stack_up
        PG_OT_stack_up = value

    def _set_export_PG_OT_stack_down(value):
        nonlocal PG_OT_stack_down
        PG_OT_stack_down = value

    def _set_export_PG_OT_stack_edit(value):
        nonlocal PG_OT_stack_edit
        PG_OT_stack_edit = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ACTIONS': lambda: ACTIONS,
        'BoolProperty': lambda: BoolProperty,
        'CONTEXT': lambda: CONTEXT,
        'CollectionProperty': lambda: CollectionProperty,
        'EnumProperty': lambda: EnumProperty,
        'IntProperty': lambda: IntProperty,
        'KINDS': lambda: KINDS,
        'PGCompositorStack': lambda: PGCompositorStack,
        'PGShaderStack': lambda: PGShaderStack,
        'PGStackEntry': lambda: PGStackEntry,
        'PGStackFocus': lambda: PGStackFocus,
        'PG_OT_stack_add': lambda: PG_OT_stack_add,
        'PG_OT_stack_down': lambda: PG_OT_stack_down,
        'PG_OT_stack_edit': lambda: PG_OT_stack_edit,
        'PG_OT_stack_remove': lambda: PG_OT_stack_remove,
        'PG_OT_stack_up': lambda: PG_OT_stack_up,
        'PointerProperty': lambda: PointerProperty,
        'SCHEMA': lambda: SCHEMA,
        'Stack': lambda: Stack,
        'StackAction': lambda: StackAction,
        'StackNode': lambda: StackNode,
        'StringProperty': lambda: StringProperty,
        'TAG': lambda: TAG,
        '_classes': lambda: _classes,
        '_write_entries': lambda: _write_entries,
        'active': lambda: active,
        'assign': lambda: assign,
        'bind_dependencies': lambda: bind_dependencies,
        'bpy': lambda: bpy,
        'changed': lambda: changed,
        'cls': lambda: cls,
        'core': lambda: core,
        'create': lambda: create,
        'draw': lambda: draw,
        'editing': lambda: editing,
        'entries': lambda: entries,
        'focus_changed': lambda: focus_changed,
        'grades': lambda: grades,
        'initialize': lambda: initialize,
        'is_stack': lambda: is_stack,
        'model': lambda: model,
        'native': lambda: native,
        'publish': lambda: publish,
        'refresh': lambda: refresh,
        'register': lambda: register,
        'repair_assignment': lambda: repair_assignment,
        'tip': lambda: tip,
        'tr': lambda: tr,
        'unregister': lambda: unregister,
        'uuid4': lambda: uuid4,
        'verify_assignment': lambda: verify_assignment,
    }, writers={
        'ACTIONS': _set_export_ACTIONS,
        'KINDS': _set_export_KINDS,
        'SCHEMA': _set_export_SCHEMA,
        'TAG': _set_export_TAG,
        '_classes': _set_export__classes,
    'core': _set_export_core, 'is_stack': _set_export_is_stack, 'active': _set_export_active, 'entries': _set_export_entries, '_write_entries': _set_export__write_entries, 'assign': _set_export_assign, 'repair_assignment': _set_export_repair_assignment, 'verify_assignment': _set_export_verify_assignment, 'model': _set_export_model, 'bind_dependencies': _set_export_bind_dependencies, 'changed': _set_export_changed, 'publish': _set_export_publish, 'refresh': _set_export_refresh, 'initialize': _set_export_initialize, 'create': _set_export_create, 'draw': _set_export_draw, 'focus_changed': _set_export_focus_changed, 'PGStackFocus': _set_export_PGStackFocus, 'PGStackEntry': _set_export_PGStackEntry, 'StackNode': _set_export_StackNode, 'PGShaderStack': _set_export_PGShaderStack, 'PGCompositorStack': _set_export_PGCompositorStack, 'StackAction': _set_export_StackAction, 'PG_OT_stack_add': _set_export_PG_OT_stack_add, 'PG_OT_stack_remove': _set_export_PG_OT_stack_remove, 'PG_OT_stack_up': _set_export_PG_OT_stack_up, 'PG_OT_stack_down': _set_export_PG_OT_stack_down, 'PG_OT_stack_edit': _set_export_PG_OT_stack_edit, 'register': _set_export_register, 'unregister': _set_export_unregister})
