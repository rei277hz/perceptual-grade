"""Short occurrence-aware source walks, independent of native scene capture."""
from dataclasses import dataclass
from ..integration.components import Component


@dataclass(frozen=True)
class Route:
    kind: str
    source: object
    steps: tuple = ()
    structure: tuple = ()
    label: str = ''


def create(application):
    import bpy
    nodes = application.component('graph.nodes')
    retain = application.component('ownership').retain

    def matching(sockets, socket):
        identifier = getattr(socket, 'identifier', '')
        return next((item for item in sockets if item.identifier == identifier), sockets.get(socket.name))

    def color(value):
        try:
            values = tuple(float(x) for x in value)
        except TypeError:
            return (float(value),) * 3 + (1.,)
        if len(values) == 4:
            return values
        if len(values) == 3:
            return values + (1.,)
        raise ValueError('The scope input cannot be represented as Color')

    def resolve(node, socket_name='Color', context=None):
        contexts = application.component('scopes.context')
        images = application.component('scopes.images')
        stacks = application.component('graph.stacks')
        frozen = contexts.source_context(node, context)
        parents = contexts.occurrence(node, frozen)
        socket = node.inputs.get(socket_name)
        if socket is None:
            raise ValueError('The scope input socket no longer exists: ' + socket_name)
        direction = 'input'
        seen, incoming, visits = set(), {}, {}
        steps, structure = [], []

        def links(value):
            tree = value.id_data
            key = tree.as_pointer()
            visits[key] = visits.get(key, 0) + 1
            if key not in incoming and visits[key] == 1:
                return tuple(link for link in value.links if not link.is_muted)
            if key not in incoming:
                table = {}
                for link in tree.links:
                    if not link.is_muted:
                        table.setdefault(link.to_socket.as_pointer(), []).append(link)
                incoming[key] = table
            return incoming[key].get(value.as_pointer(), ())

        def result(kind, source, label):
            return Route(kind, source, tuple(reversed(steps)), tuple(structure), label)

        try:
            while socket is not None:
                occurrence = tuple(parent.as_pointer() for parent in parents)
                identity = (direction, socket.as_pointer(), occurrence)
                if identity in seen:
                    raise ValueError('Scope source has a cycle')
                seen.add(identity)
                if direction == 'input':
                    incoming_links = links(socket)
                    if not incoming_links:
                        if not hasattr(socket, 'default_value'):
                            raise ValueError('The scope source is unconnected')
                        return result('constant', color(socket.default_value), 'Node input color')
                    if len(incoming_links) != 1:
                        return result('native', None, 'Processed Blender input')
                    socket = incoming_links[0].from_socket
                    direction = 'output'
                    continue
                source = socket.node
                kind = source.bl_idname
                if source.mute:
                    bypass = [link.from_socket for link in source.internal_links if link.to_socket == socket]
                    if len(bypass) == 1:
                        socket, direction = bypass[0], 'input'
                        continue
                    return result('native', None, 'Muted Blender input')
                if kind == 'NodeReroute':
                    socket, direction = source.inputs[0], 'input'
                    continue
                if kind == 'NodeGroupInput':
                    if not parents:
                        raise ValueError('Choose the node-group instance to sample in the node editor')
                    socket = matching(parents[-1].inputs, socket)
                    parents, direction = parents[:-1], 'input'
                    continue
                is_grade, is_stack = nodes.is_grade(source), stacks.is_stack(source)
                # Recovery adopts Blender's authored output without claiming
                # that any output still implements the saved analytical model.
                if (is_grade or is_stack) and application.component('graph.lut').is_adopted(source):
                    structure.append((kind, source.as_pointer(), occurrence, 'adopted'))
                    return result('native', None, 'Restored Blender input')
                if (is_grade or is_stack) and socket.name in {'Baked Color', 'Baked Alpha'}:
                    image = application.component('graph.bake').baked_image(source)
                    if image is None:
                        return result('constant', (0., 0., 0., 0.), 'Unbound baked image')
                    alpha = socket.name == 'Baked Alpha'
                    kind = str(image.get('_pg_bake_kind', 'COMPOSITOR'))
                    descriptor = images.describe(image,
                        source_kind='shader' if kind.startswith('SHADER_') else 'compositor',
                        channel='ALPHA' if alpha else 'COLOR',
                        premultiply=(kind == 'COMPOSITOR' and not alpha))
                    return result('image', descriptor,
                                  ('Baked alpha: ' if alpha else 'Baked image: ') + image.name)
                if is_grade or is_stack:
                    if socket.name == 'Alpha':
                        socket, direction = source.inputs.get('Alpha'), 'input'
                        continue
                    if socket.name != 'Color':
                        return result('native', None, 'Processed Blender input')
                    handle = retain(source, 'scope.source_model')
                    steps.append(('stack' if is_stack else 'grade', handle))
                    structure.append((kind, source.as_pointer(), occurrence))
                    socket, direction = source.inputs.get('Color'), 'input'
                    continue
                if kind in {'ShaderNodeGroup', 'CompositorNodeGroup', 'PGShaderBake', 'PGCompositorBake'}:
                    child = getattr(source, 'node_tree', None)
                    if child is not None and (child == source.id_data or any(
                            child == parent.id_data for parent in parents)):
                        raise ValueError('Scope source has a recursive node group')
                    outputs = tuple(item for item in child.nodes if item.bl_idname == 'NodeGroupOutput') if child else ()
                    active = next((item for item in outputs if item.is_active_output), outputs[0] if outputs else None)
                    if active is None:
                        raise ValueError('Scope source group has no active output')
                    socket = matching(active.inputs, socket)
                    parents, direction = parents + (source,), 'input'
                    continue
                if kind == 'CompositorNodeTransform':
                    socket, direction = source.inputs.get('Image'), 'input'
                    continue
                if kind in {'ShaderNodeRGB', 'CompositorNodeRGB', 'ShaderNodeValue', 'CompositorNodeValue'}:
                    return result('constant', color(socket.default_value), 'Connected color')
                if kind in {'ShaderNodeTexImage', 'CompositorNodeImage'} and source.image is not None:
                    image = source.image
                    if image.source == 'VIEWER' or image.type in {'RENDER_RESULT', 'COMPOSITING'}:
                        return result('native', None, 'Rendered Blender input')
                    # A null-ImageUser size query can reload a static multiview
                    # Image and discard its resident painted-eye buffers.
                    if not image.use_multiview:
                        _dimensions = tuple(image.size)
                    user = getattr(source, 'image_user', source)
                    scene = bpy.data.scenes.get(frozen.get('scene', '')) or bpy.context.scene
                    frame = int(scene.frame_current) if scene else 1
                    settings = {name: getattr(user, name, default) for name, default in (
                        ('frame_start', 1), ('frame_duration', 1), ('frame_offset', 0), ('use_cyclic', False))}
                    compositor = kind == 'CompositorNodeImage'
                    alpha_linked = bool(source.outputs.get('Alpha') and source.outputs['Alpha'].is_linked)
                    # Pass selection is a source property; ordinary Alpha outputs
                    # remain native until their scalar conversion is represented.
                    descriptor = images.describe(image, source_kind='compositor' if compositor else 'shader',
                        frame_settings=settings, scene_frame=frame, scene=scene,channel='ALPHA' if socket.name=='Alpha' else 'COLOR',
                        layer=(getattr(source, 'layer', None) if compositor else getattr(user, 'multilayer_layer', None)) if image.type=='MULTILAYER' else None,
                        pass_name=('Combined' if socket.name in {'Image', 'Color'} else socket.name) if compositor
                                  else getattr(user, 'multilayer_pass', 0),
                        view=(getattr(source, 'view', None) if compositor else getattr(user, 'multilayer_view', None)) if image.use_multiview else None,
                        premultiply=compositor or not alpha_linked)
                    return result('image', descriptor, 'Input image: ' + image.name)
                return result('native', None, 'Processed Blender input')
            raise ValueError('The scope source socket is unavailable')
        finally:
            incoming.clear()
            seen.clear()
            visits.clear()

    return Component({'__name__': __name__, 'Route': Route, 'resolve': resolve})
