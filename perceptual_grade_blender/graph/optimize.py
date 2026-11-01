"""Small native-graph reductions, independent of grading model and UI.

Apply folding only after the whole tree is wired: this makes single-use
analysis reliable and does not invalidate an in-progress builder's CSE cache.
"""

from ..integration.components import Component

def create(_application):
    import struct
    remove_node = _application.app.blender.native.remove_node


    def f32(value):
        """Round an internal uniform as Blender's float socket would."""
        return struct.unpack('f', struct.pack('f', float(value)))[0]


    def fold_multiply_adds(tree):
        """Replace single-use multiply/add pairs with a native Multiply Add node.

        No reassociation, subtraction rewriting, validation removal or clamping.
        Backend rounding may nevertheless differ; render regression tests cover it.
        """
        users = {}
        for link in tree.links:
            key = link.from_node.as_pointer()
            users[key] = users.get(key, 0) + 1
        candidates = []
        for node in tree.nodes:
            if node.bl_idname != 'ShaderNodeMath' or node.operation != 'ADD' or node.use_clamp or node.mute:
                continue
            for index in (0, 1):
                socket = node.inputs[index]
                if not socket.is_linked:
                    continue
                mul = socket.links[0].from_node
                if (mul.bl_idname == 'ShaderNodeMath' and mul.operation == 'MULTIPLY'
                        and not mul.use_clamp and not mul.mute
                        and users.get(mul.as_pointer(), 0) == 1):
                    candidates.append((node.name, mul.name, index))
                    break

        def read(socket):
            if not socket.is_linked:
                return ('value', float(socket.default_value))
            source = socket.links[0].from_socket
            return ('socket', source.node.name, list(source.node.outputs).index(source))

        for node_name, mul_name, index in candidates:
            node, mul = tree.nodes[node_name], tree.nodes[mul_name]
            values = (read(mul.inputs[0]), read(mul.inputs[1]), read(node.inputs[1-index]))
            for socket in node.inputs:
                for link in tuple(socket.links):
                    tree.links.remove(link)
            node.operation = 'MULTIPLY_ADD'
            node.label = 'Multiply Add'
            for socket, value in zip(node.inputs, values):
                if value[0] == 'socket':
                    tree.links.new(tree.nodes[value[1]].outputs[value[2]], socket)
                else:
                    socket.default_value = value[1]
            remove_node(tree, mul)
            mul = None
        tree['_pg_fused_madd'] = len(candidates)
        return len(candidates)


    def remove_dead_nodes(tree):
        """Discard unused forward attributes/helper outputs after specialization."""
        parents = {}
        for link in tree.links:
            parents.setdefault(link.to_node.as_pointer(), set()).add(link.from_node.as_pointer())
        keep = set()
        pending = [node.as_pointer() for node in tree.nodes
                   if node.type == 'GROUP_OUTPUT' and node.is_active_output]
        while pending:
            key = pending.pop()
            if key in keep:
                continue
            keep.add(key)
            pending.extend(parents.get(key, ()))
        # Keep native wrapper collections out of the destruction loop.
        names = tuple(node.name for node in tree.nodes
                      if node.type != 'GROUP_INPUT' and node.as_pointer() not in keep)
        for name in names:
            node = tree.nodes[name]
            remove_node(tree, node)
            node = None

    def _set_export_f32(value):
        nonlocal f32
        f32 = value

    def _set_export_fold_multiply_adds(value):
        nonlocal fold_multiply_adds
        fold_multiply_adds = value

    def _set_export_remove_dead_nodes(value):
        nonlocal remove_dead_nodes
        remove_dead_nodes = value

    return Component(readers={
        "__name__": lambda: __name__,
        'f32': lambda: f32,
        'fold_multiply_adds': lambda: fold_multiply_adds,
        'remove_dead_nodes': lambda: remove_dead_nodes,
        'remove_node': lambda: remove_node,
        'struct': lambda: struct,
    }, writers={
    'f32': _set_export_f32, 'fold_multiply_adds': _set_export_fold_multiply_adds, 'remove_dead_nodes': _set_export_remove_dead_nodes})
