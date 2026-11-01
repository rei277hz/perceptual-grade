"""Native removals retire only references belonging to their explicit App."""
from ..ownership import OwnershipContext


def keys(value):
    import bpy
    if not isinstance(value, bpy.types.bpy_struct):
        return None, None, None
    identity = int(value.as_pointer())
    owner = value.id_data
    root = None if owner is None else int(owner.as_pointer())
    parent = int(value.node.as_pointer()) if isinstance(value, bpy.types.NodeSocket) else None
    return identity, root, parent


class NativeAdapter:
    def __init__(self, ownership):
        if not isinstance(ownership, OwnershipContext):
            raise TypeError('Native adapter requires an explicit ownership context')
        self.ownership = ownership

    def remove_node(self, tree, node):
        self.ownership.retire(node)
        tree.nodes.remove(node)

    def remove_image(self, image):
        import bpy
        self.ownership.retire_datablock(image)
        bpy.data.images.remove(image)

    def remove_node_group(self, group):
        import bpy
        self.ownership.retire_datablock(group)
        bpy.data.node_groups.remove(group)

    def remove_action(self, action):
        import bpy
        self.ownership.retire_datablock(action)
        bpy.data.actions.remove(action)

    def remove_timer(self, window_manager, timer):
        self.ownership.retire(timer)
        window_manager.event_timer_remove(timer)

    def remove_keymap_item(self, keymap, item):
        self.ownership.retire(item)
        keymap.keymap_items.remove(item)
