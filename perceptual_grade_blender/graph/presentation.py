"""Presentation for owned native groups; ordinary user groups stay untouched."""

from ..integration.components import Component

def create(_application):
    OWNED_KEYS=('_perceptual_grade_kernel','_pg_native_op','_pg_cube_op')


    def private_name(name):
        return name if name.startswith('.') else '.'+name


    def hide_internal_groups():
        # Called only after unrestricted data access, including on load and re-enable.
        import bpy
        if not hasattr(bpy.data,'node_groups'):return
        for tree in bpy.data.node_groups:
            if tree.library is None and any(tree.get(key) for key in OWNED_KEYS):
                if not tree.name.startswith('.'):tree.name=private_name(tree.name)


    def node_options(node):
        KINDS = _application.component('graph.grades').KINDS
        STACKS = _application.component('graph.stacks').KINDS
        show = node.bl_idname in (*KINDS.values(),*STACKS.values())
        # Even assigning the existing RNA value dirties the owning node tree.
        if node.show_options != show:
            node.show_options = show

    def _set_export_OWNED_KEYS(value):
        nonlocal OWNED_KEYS
        OWNED_KEYS = value

    def _set_export_private_name(value):
        nonlocal private_name
        private_name = value

    def _set_export_hide_internal_groups(value):
        nonlocal hide_internal_groups
        hide_internal_groups = value

    def _set_export_node_options(value):
        nonlocal node_options
        node_options = value

    return Component(readers={
        "__name__": lambda: __name__,
        'OWNED_KEYS': lambda: OWNED_KEYS,
        'hide_internal_groups': lambda: hide_internal_groups,
        'node_options': lambda: node_options,
        'private_name': lambda: private_name,
    }, writers={
        'OWNED_KEYS': _set_export_OWNED_KEYS,
    'private_name': _set_export_private_name, 'hide_internal_groups': _set_export_hide_internal_groups, 'node_options': _set_export_node_options})
