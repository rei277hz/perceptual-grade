"""Value-only route discovery for Perceptual Grade shader baking."""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import os
from array import array

from ..integration.components import Component


class RouteBlocked(ValueError):
    """An expected input condition, shared by observation and preparation."""
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def create(application):
    import bpy

    contexts = application.component('scopes.context')
    image_sources = application.component('scopes.images')
    nodes = application.component('graph.nodes')
    stacks = application.component('graph.stacks')
    lut = application.component('graph.lut')
    bake = application.component('graph.bake')

    SCHEMA = 1
    POINTWISE = frozenset({
        'ShaderNodeRGB', 'ShaderNodeValue', 'ShaderNodeRGBToBW',
        'ShaderNodeMath', 'ShaderNodeVectorMath', 'ShaderNodeMix',
        'ShaderNodeMixRGB', 'ShaderNodeHueSaturation', 'ShaderNodeBrightContrast',
        'ShaderNodeGamma', 'ShaderNodeInvert', 'ShaderNodeValToRGB',
        'ShaderNodeRGBCurve', 'ShaderNodeCombineColor', 'ShaderNodeSeparateColor',
        'ShaderNodeCombineRGB', 'ShaderNodeSeparateRGB', 'ShaderNodeClamp',
        'ShaderNodeMapRange', 'ShaderNodeMapValue',
    })
    IMAGE_NODE = 'ShaderNodeTexImage'
    GROUP_NODE = 'ShaderNodeGroup'

    def _pointer(value):
        try:
            return int(value.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError):
            return 0

    def _plain(value):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, Mapping):
            return {str(key): _plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(_plain(item) for item in value)
        try:
            return tuple(_plain(item) for item in value)
        except (TypeError, ValueError):
            return str(value)

    def _rna_state(value, seen=None, depth=0):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if depth >= 6:
            return ('depth', type(value).__name__)
        try:
            pointer = _pointer(value)
            if pointer:
                seen = set() if seen is None else seen
                if pointer in seen:
                    return ('reference', type(value).__name__)
                seen = seen | {pointer}
            properties = value.bl_rna.properties
        except (AttributeError, ReferenceError, RuntimeError):
            return _plain(value)
        ignored = {'rna_type', 'id_data', 'original', 'name', 'name_full',
                   'evaluated_get', 'path_from_id', 'inputs', 'outputs', 'internal_links',
                   'node_tree', 'links', 'parent'}
        result = []
        for prop in properties:
            name = str(prop.identifier)
            if name in ignored or (prop.is_readonly and prop.type not in {'POINTER', 'COLLECTION'}):
                continue
            try:
                item = getattr(value, name)
                if prop.type == 'COLLECTION':
                    frozen = tuple(_rna_state(entry, seen, depth + 1) for entry in item)
                elif prop.type == 'POINTER' and item is not None:
                    if isinstance(item, bpy.types.ID):
                        frozen = (item.bl_rna.identifier, str(item.name_full), int(item.session_uid))
                    else:
                        frozen = _rna_state(item, seen, depth + 1)
                else:
                    frozen = _plain(item)
                result.append((name, frozen))
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                continue
        return tuple(result)

    def _id_key(value):
        if value is None:
            return None
        return str(value.bl_rna.identifier), str(value.name_full), int(value.session_uid)

    def _image_state(image, scene):
        descriptor = image_sources.describe(image, source_kind='shader', scene=scene)
        path = bpy.path.abspath(image.filepath, library=image.library) if image.filepath else ''
        try:
            stat = os.stat(path)
            resource = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            resource = None
        packed = tuple((entry.tile_number, entry.view, entry.packed_file.size)
                       for entry in getattr(image, 'packed_files', ()))
        pixels = None
        if bool(image.is_dirty) or str(image.source) == 'GENERATED':
            width, height = (int(value) for value in image.size)
            count = width * height * int(image.channels)
            if count < 1 or count * 4 > 512 * 1024 * 1024:
                raise RouteBlocked('dimensions', 'Shader Bake source Image exceeds the 512 MiB pixel budget')
            if not image.has_data:
                raise RouteBlocked('source_empty', 'Shader Bake source Image has no loaded pixels')
            values = array('f', [0.]) * count
            image.pixels.foreach_get(values)
            pixels = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        return (_plain(descriptor.key), _plain(descriptor.metadata), bool(image.is_dirty), packed,
                int(image.size[0]), int(image.size[1]), bool(image.is_float), resource, pixels)

    def _node_fingerprint(source, output, path):
        semantic = application.component('graph.lut').scope_signature(source) if _is_grade(source) else _rna_state(source)
        sockets = tuple((socket.identifier, _plain(getattr(socket, 'default_value', None)),
                         tuple((link.from_node.name, link.from_socket.identifier, bool(link.is_muted))
                               for link in _links(socket))) for socket in source.inputs)
        return (tuple(path), str(source.name), str(source.bl_idname), str(output.identifier),
                _plain(semantic), sockets)

    def _parameter_observation(node):
        grades = application.component('graph.grades')
        if stacks.is_stack(node):
            return tuple((_id_key(grade), enabled, direction, key,
                          repr(_plain(grades.snapshot(grade))) if grades.is_data(grade) else None)
                         for grade, enabled, direction, key, _focus in stacks.entries(node))
        return repr(_plain(grades.snapshot(grades.data_owner(node))))

    def _node_observation(source, output, path):
        values = _parameter_observation(source) if _is_grade(source) else _rna_state(source)
        sockets = tuple((socket.identifier, _plain(getattr(socket, 'default_value', None)),
                         tuple((link.from_node.name, link.from_socket.identifier, bool(link.is_muted))
                               for link in _links(socket))) for socket in source.inputs)
        return (tuple(path), str(source.name), str(source.bl_idname), str(output.identifier),
                _plain(values), sockets)

    def _links(socket):
        return tuple(link for link in socket.links if not link.is_muted)

    def _single_link(socket):
        links = _links(socket)
        if len(links) > 1:
            raise RouteBlocked('input_links', 'Shader Bake input has multiple links')
        return links[0] if links else None

    def _owner(node, context):
        frozen = context if isinstance(context, Mapping) and not context.get('_lightweight') else None
        if frozen is not None:
            owner_kind, owner_name = frozen.get('owner_kind'), frozen.get('owner_name')
            scene = bpy.data.scenes.get(str(frozen.get('scene', '')))
            owner = (bpy.data.materials if owner_kind == 'material' else bpy.data.worlds).get(str(owner_name))
            if owner is None or scene is None:
                raise ReferenceError('Shader Bake owner or scene is unavailable')
            return owner_kind, owner, scene, tuple(frozen.get('group_path', ()))

        context = context or bpy.context
        scene = getattr(context, 'scene', None) or bpy.context.scene
        space = getattr(context, 'space_data', None)
        editor_owner = getattr(space, 'id', None)
        if editor_owner is not None:
            if isinstance(editor_owner, bpy.types.Material):
                kind = 'material'
            elif isinstance(editor_owner, bpy.types.World):
                kind = 'world'
            else:
                kind = ''
            root = getattr(editor_owner, 'node_tree', None)
            if kind and root is not None and contexts.group_paths(root, node.id_data):
                if kind == 'world' and editor_owner != scene.world:
                    raise RouteBlocked('world_scene', 'The selected World is not used by the current scene')
                path = contexts.editor_path(space, root, node.id_data)
                if path is None:
                    paths = contexts.group_paths(root, node.id_data)
                    if len(paths) != 1:
                        raise RouteBlocked('occurrence', 'Choose the exact node-group occurrence to bake')
                    path = paths[0]
                return kind, editor_owner, scene, tuple(path)
        candidates = []
        for owner_kind, collection in (('material', bpy.data.materials), ('world', bpy.data.worlds)):
            for owner in collection:
                root = getattr(owner, 'node_tree', None)
                if root is not None and contexts.group_paths(root, node.id_data):
                    candidates.append((owner_kind, owner, root))

        def matches_owner(candidate, value):
            return value is not None and _pointer(candidate[1]) == _pointer(value)

        selected = [item for item in candidates if matches_owner(item, editor_owner)]
        if not selected:
            obj = getattr(context, 'object', None)
            material = getattr(obj, 'active_material', None)
            selected = [item for item in candidates if matches_owner(item, material)]
        if not selected and scene is not None:
            selected = [item for item in candidates if item[0] == 'world' and item[1] == scene.world]
        if not selected:
            selected = candidates
        if len(selected) != 1:
            raise RouteBlocked('owner', 'Choose the material or world containing this shader node')
        owner_kind, owner, root = selected[0]
        if owner_kind == 'world' and owner != scene.world:
            raise RouteBlocked('world_scene', 'The selected World is not used by the current scene')
        path = contexts.editor_path(space, root, node.id_data)
        if path is None:
            paths = contexts.group_paths(root, node.id_data)
            if len(paths) != 1:
                raise RouteBlocked('occurrence', 'Choose the exact node-group occurrence to bake')
            path = paths[0]
        return owner_kind, owner, scene, tuple(path)

    def _image(source, output_name, path, scene, *, fingerprint=True):
        image = getattr(source, 'image', None)
        if image is None:
            raise RouteBlocked('source_unbound', 'Shader Bake does not support an unbound Image Texture')
        source_kind = str(image.source)
        if source_kind in {'MOVIE', 'SEQUENCE'}:
            raise RouteBlocked('source_sequence', 'Shader Bake does not support movie or image-sequence sources')
        if source_kind == 'TILED' or bool(getattr(image, 'is_tiled', False)):
            raise RouteBlocked('source_udim', 'Shader Bake does not support UDIM sources')
        if source_kind not in {'FILE', 'GENERATED'}:
            raise RouteBlocked('source_type', 'Shader Bake requires a stored still Image Texture')
        if image.use_multiview:
            raise RouteBlocked('source_multiview', 'Shader Bake does not support multiview Image Texture sources')
        if image.type not in {'IMAGE', 'UV_TEST'}:
            raise RouteBlocked('source_multilayer', 'Shader Bake does not support multilayer Image Texture sources')
        if not fingerprint:
            return dict(group_path=tuple(path), node_name=str(source.name), image_name=str(image.name),
                        alpha_linked=output_name == 'Alpha',
                        image_source=source_kind,
                        width=image.get('_pg_bake_width'), height=image.get('_pg_bake_height'))
        width, height = (int(value) for value in image.size)
        if width < 1 or height < 1:
            raise RouteBlocked('source_empty', 'Shader Bake Image Texture has no pixels')
        if width * height * max(4, int(image.channels)) * 4 > 512 * 1024 * 1024:
            raise RouteBlocked('dimensions', 'Shader Bake source Image exceeds the 512 MiB pixel budget')
        user = getattr(source, 'image_user', source)
        return dict(group_path=tuple(path), node_name=str(source.name), image_name=str(image.name),
                    alpha_linked=output_name == 'Alpha',
                    width=width, height=height, image_source=source_kind,
                    colorspace=str(image.colorspace_settings.name), alpha_mode=str(image.alpha_mode),
                    interpolation=str(source.interpolation), extension=str(source.extension),
                    projection=str(source.projection), frame_start=int(user.frame_start),
                    frame_duration=int(user.frame_duration), frame_offset=int(user.frame_offset),
                    use_cyclic=bool(user.use_cyclic),
                    **({'image_state': _image_state(image, scene)} if fingerprint else {}))

    def _kernel_image(node):
        return bake.baked_image(node)

    def _is_grade(node):
        return bool(nodes.is_grade(node) or stacks.is_stack(node))

    def _require_uv(state, reason='surface_dependency'):
        state['native'] = True
        # A coordinate branch already has the more useful linked-coordinates
        # explanation. Do not also describe its UV Map/TexCoord nodes as a
        # separate surface-dependent color computation.
        if reason != 'surface_dependency' or not state['coordinate_depth']:
            state['reasons'].add(reason)

    def _resolve_group_input(source, output, path, seen, images, upstream, state):
        if not path:
            _require_uv(state)
            return
        owner = state['owner']
        root = owner.node_tree
        for name in path[:-1]:
            group = root.nodes.get(name)
            if group is None or group.node_tree is None:
                raise ReferenceError('Shader Bake group occurrence changed')
            root = group.node_tree
        group = root.nodes.get(path[-1])
        if group is None:
            raise ReferenceError('Shader Bake group occurrence changed')
        parent_input = next((socket for socket in group.inputs
                             if socket.identifier == output.identifier or socket.name == output.name), None)
        if parent_input is None:
            _require_uv(state)
            return
        _walk_input(parent_input, path[:-1], seen, images, upstream, state)

    def _walk_output(source, output, path, seen, images, upstream, state):
        key = (_pointer(source), str(output.identifier), tuple(path))
        if key in seen:
            raise RouteBlocked('cycle', 'Shader Bake source contains a cycle')
        seen.add(key)
        try:
            kind = str(source.bl_idname)
            # Node settings and links are scalar observations. Image content is
            # inspected separately only by the explicit fingerprinting path.
            state['graph'].append((_node_fingerprint if state.get('fingerprint', True)
                                   else _node_observation)(source, output, path))
            if source.mute:
                bypass = tuple(link.from_socket for link in source.internal_links if link.to_socket == output)
                if len(bypass) == 1:
                    _walk_output(bypass[0].node, bypass[0], path, seen, images, upstream, state)
                else:
                    _require_uv(state)
                return
            if kind == 'ShaderNodeShaderToRGB':
                raise RouteBlocked('shader_to_rgb', 'Shader Bake does not support Shader to RGB')
            if kind == IMAGE_NODE:
                images.append(_image(source, str(output.name), path, state['scene'],
                                     fingerprint=state.get('fingerprint', True)))
                # The Image Texture vector input controls placement, so it is
                # retained by the UV route and cannot enter source-image mode.
                if _links(source.inputs['Vector']):
                    _require_uv(state, 'linked_coordinates')
                    state['coordinate_depth'] += 1
                    try:
                        _walk_input(source.inputs['Vector'], path, seen, images, upstream, state)
                    finally:
                        state['coordinate_depth'] -= 1
                return
            if kind == 'NodeGroupInput':
                _resolve_group_input(source, output, path, seen, images, upstream, state)
                return
            if kind == 'NodeReroute':
                if source.inputs:
                    _walk_input(source.inputs[0], path, seen, images, upstream, state)
                return
            if kind == GROUP_NODE and not _is_grade(source):
                child = source.node_tree
                if child is None:
                    raise RouteBlocked('group_tree', 'Shader Bake group has no node tree')
                output_node = next((item for item in child.nodes if item.bl_idname == 'NodeGroupOutput'
                                    and item.is_active_output), None)
                if output_node is None:
                    raise RouteBlocked('group_output', 'Shader Bake group has no active output')
                child_input = next((item for item in output_node.inputs
                                    if item.identifier == output.identifier or item.name == output.name), None)
                if child_input is None:
                    raise RouteBlocked('group_output', 'Shader Bake group output is unavailable')
                _walk_input(child_input, path + (str(source.name),), seen, images, upstream, state)
                return
            if _is_grade(source):
                if output.name in {'Baked Color', 'Baked Alpha'}:
                    image = _kernel_image(source)
                    if image is None:
                        raise RouteBlocked('baked_unbound', 'Shader Bake Baked Color output is unbound')
                    images.append(dict(group_path=path + (str(source.name),),
                                       node_name='Perceptual Grade Baked Image',
                                       image_name=str(image.name), alpha_linked=output.name == 'Baked Alpha',
                                       width=(int(image.size[0]) if state.get('fingerprint', True)
                                              else image.get('_pg_bake_width')),
                                       height=(int(image.size[1]) if state.get('fingerprint', True)
                                               else image.get('_pg_bake_height')),
                                       image_source='BAKED',
                                       **({'image_state': _image_state(image, state['scene'])}
                                          if state.get('fingerprint', True) else {})))
                    return
                if output.name == 'Alpha':
                    alpha = source.inputs.get('Alpha')
                    if alpha is not None:
                        _walk_input(alpha, path, seen, images, upstream, state)
                    return
                if lut.is_adopted(source):
                    _require_uv(state)
                    state['adopted'] = True
                    state['intent_upstream'].append(dict(
                        kind='STACK' if stacks.is_stack(source) else 'GRADE',
                        group_path=tuple(path), node_name=str(source.name)))
                    return
                state['intent_upstream'].append(dict(
                    kind='STACK' if stacks.is_stack(source) else 'GRADE',
                    group_path=tuple(path), node_name=str(source.name)))
                upstream.append(dict(kind='STACK' if stacks.is_stack(source) else 'GRADE',
                                     group_path=tuple(path), node_name=str(source.name)))
                input_names = ('Alpha',) if output.name == 'Alpha' else ('Color', 'Alpha')
                for input_name in input_names:
                    socket = source.inputs.get(input_name)
                    if socket is not None:
                        _walk_input(socket, path, seen, images, upstream, state)
                return
            if kind not in POINTWISE:
                _require_uv(state)
            for socket in source.inputs:
                if kind == IMAGE_NODE and socket.name == 'Vector':
                    continue
                _walk_input(socket, path, seen, images, upstream, state)
        finally:
            seen.remove(key)

    def _walk_input(socket, path, seen, images, upstream, state):
        link = _single_link(socket)
        if link is not None:
            _walk_output(link.from_node, link.from_socket, path, seen, images, upstream, state)

    def _dependencies(node, group_path, owner, scene, *, fingerprint=True):
        seen, images, upstream = set(), [], []
        state = {'native': False, 'adopted': False, 'owner': owner, 'scene': scene, 'graph': [],
                 'intent_upstream': [],
                 'fingerprint': bool(fingerprint), 'reasons': set(), 'coordinate_depth': 0}
        # Color and Alpha are separate inputs in the product kernel. Walking both
        # prevents an unsupported alpha-side source from being silently skipped.
        for name in ('Color', 'Alpha'):
            socket = node.inputs.get(name)
            if socket is not None:
                _walk_input(socket, tuple(group_path), seen, images, upstream, state)
        unique = {}
        for item in images:
            key = (tuple(item['group_path']), item['node_name'], item['image_name'])
            previous = unique.get(key)
            if previous is None:
                unique[key] = item
            else:
                previous['alpha_linked'] = previous['alpha_linked'] or item['alpha_linked']
        graph = tuple(sorted(set(state['graph']), key=repr))
        ordered, seen_upstream = [], set()
        for item in reversed(upstream):
            key = (item['kind'], tuple(item['group_path']), item['node_name'])
            if key not in seen_upstream:
                seen_upstream.add(key)
                ordered.append(item)
        return tuple(unique.values()), tuple(ordered), graph, state

    def _active_object(context, scene):
        if isinstance(context, Mapping):
            object_name = context.get('shader_bake', {}).get('object_name', '')
            if 'shader_bake' in context:
                return bpy.data.objects.get(object_name) if object_name else None
            context = bpy.context
        obj = getattr(context, 'object', None)
        if obj is not None:
            return obj
        view_layer = getattr(context, 'view_layer', None)
        obj = getattr(getattr(view_layer, 'objects', None), 'active', None)
        if obj is not None:
            return obj
        return None

    def _uv_triangles(mesh, uv_layer):
        mesh.calc_loop_triangles()
        result = []
        for triangle in mesh.loop_triangles:
            points = tuple(tuple(float(value) for value in uv_layer.data[index].uv)
                           for index in triangle.loops)
            area = abs(sum(points[i][0] * points[(i + 1) % 3][1]
                           - points[(i + 1) % 3][0] * points[i][1]
                           for i in range(3))) * .5
            if area <= 1e-12:
                raise RouteBlocked('uv_degenerate', 'Shader UV Bake UV map contains a degenerate face')
            result.append(points)
        if not result:
            raise RouteBlocked('uv_empty', 'Shader UV Bake mesh has no UV-mapped faces')
        return tuple(result)

    def _intersection_area(subject, clip):
        orientation = sum(clip[i][0] * clip[(i + 1) % 3][1]
                          - clip[(i + 1) % 3][0] * clip[i][1] for i in range(3))
        sign = 1. if orientation >= 0. else -1.
        polygon = list(subject)
        for index in range(3):
            ax, ay = clip[index]
            bx, by = clip[(index + 1) % 3]
            previous = polygon
            polygon = []
            if not previous:
                return 0.
            prior = previous[-1]
            prior_side = sign * ((bx - ax) * (prior[1] - ay) - (by - ay) * (prior[0] - ax))
            for current in previous:
                side = sign * ((bx - ax) * (current[1] - ay) - (by - ay) * (current[0] - ax))
                if (side >= -1e-12) != (prior_side >= -1e-12):
                    dx, dy = current[0] - prior[0], current[1] - prior[1]
                    edge_x, edge_y = bx - ax, by - ay
                    denominator = edge_x * dy - edge_y * dx
                    if abs(denominator) > 1e-20:
                        t = (edge_x * (ay - prior[1]) - edge_y * (ax - prior[0])) / denominator
                        polygon.append((prior[0] + t * dx, prior[1] + t * dy))
                if side >= -1e-12:
                    polygon.append(current)
                prior, prior_side = current, side
        return abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                       - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                       for i in range(len(polygon)))) * .5 if len(polygon) >= 3 else 0.

    def _validate_uv_nonoverlap(triangles):
        bins = {}
        for index, points in enumerate(triangles):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            cells = {(x, y) for x in range(max(0, int(min(xs) * 32)), min(31, int(max(xs) * 32)) + 1)
                     for y in range(max(0, int(min(ys) * 32)), min(31, int(max(ys) * 32)) + 1)}
            candidates = set()
            for cell in cells:
                candidates.update(bins.get(cell, ()))
            if len(candidates) > 250000:
                raise RouteBlocked('uv_complex', 'Shader UV Bake UV overlap check exceeded its bounded work limit')
            for other in candidates:
                left, right = triangles[other], points
                if (max(p[0] for p in left) <= min(xs) + 1e-12
                        or max(xs) <= min(p[0] for p in left) + 1e-12
                        or max(p[1] for p in left) <= min(ys) + 1e-12
                        or max(ys) <= min(p[1] for p in left) + 1e-12):
                    continue
                if _intersection_area(left, right) > 1e-10:
                    raise RouteBlocked('uv_overlap', 'Shader UV Bake UV map contains overlapping faces')
            for cell in cells:
                bins.setdefault(cell, []).append(index)

    def _uv(obj, *, inspect_content=True):
        if obj is None or obj.type != 'MESH' or obj.data is None:
            raise RouteBlocked('mesh', 'Shader UV Bake requires an active mesh object')
        layers = obj.data.uv_layers
        uv_layer = getattr(layers, 'active_render', None)
        if uv_layer is None:
            raise RouteBlocked('uv_missing', 'Shader UV Bake requires an active render UV map')
        if not uv_layer.data or len(uv_layer.data) != len(obj.data.loops):
            raise RouteBlocked('uv_incomplete', 'Shader UV Bake active render UV map is incomplete')
        if inspect_content:
            coordinates = tuple(float(value) for item in uv_layer.data for value in item.uv)
            if any(not (-1e-7 <= value <= 1.0000001) for value in coordinates):
                raise RouteBlocked('uv_tile', 'Shader UV Bake supports only the 0-1 UV tile')
            _validate_uv_nonoverlap(_uv_triangles(obj.data, uv_layer))
        if any(getattr(modifier, 'type', '') in {'MESH_SEQUENCE_CACHE', 'MESH_CACHE'}
               or getattr(modifier, 'cache_file', None) is not None for modifier in obj.modifiers):
            raise RouteBlocked('mesh_cache', 'Shader UV Bake does not support Alembic or CacheFile geometry')
        return str(uv_layer.name)

    def _material_slot(obj, material):
        matches = tuple(index for index, slot in enumerate(obj.material_slots) if slot.material == material)
        if not matches:
            raise RouteBlocked('material_slot', 'Active mesh does not use the selected material')
        active = int(getattr(obj, 'active_material_index', matches[0]))
        return active if active in matches else matches[0]

    def _bound_image(node):
        """Return the persisted bake binding without reading image pixels."""
        try:
            image = bake.baked_image(node)
        except (AttributeError, ReferenceError, RuntimeError):
            return None, ''
        if image is None:
            return None, ''
        return image, str(image.get('_pg_bake_kind', ''))

    def _route_domain(node, state, images, bound_kind):
        graph_domain = 'IMAGE' if not state['native'] and len(images) <= 1 else 'UV'
        if bound_kind == 'SHADER_UV':
            # A UV result remains a UV result even if an edit makes its current
            # source graph look image-addressable. Reusing it as an image-domain
            # artifact would silently change the meaning of the shared binding.
            return 'UV'
        return graph_domain

    def _destination_compatible(domain, bound_kind):
        if domain == 'UV' and bound_kind in {'SHADER_IMAGE', 'COMPOSITOR'}:
            raise RouteBlocked('destination_domain',
                'The bound baked Image is Image-domain or Compositor data, but this graph now requires '
                'a UV Shader Bake; clear or replace the binding before baking')

    def _size(node, domain, images, bound_image=None, *, passive=False):
        if domain == 'IMAGE':
            if not images:
                return (1, 1)
            if passive and (images[0]['width'] is None or images[0]['height'] is None):
                return None, None
            width, height = int(images[0]['width']), int(images[0]['height'])
            if width * height * 16 > 512 * 1024 * 1024:
                raise RouteBlocked('dimensions', 'Shader Image Bake dimensions exceed the 512 MiB pixel budget')
            return width, height
        if bound_image is not None and bound_image.get('_pg_bake_kind') != 'EMPTY':
            if passive:
                width, height = int(bound_image.get('_pg_bake_width', 0)), int(bound_image.get('_pg_bake_height', 0))
            else:
                width, height = (int(value) for value in bound_image.size)
        else:
            width = int(node.get('_pg_bake_width', 2048))
            height = int(node.get('_pg_bake_height', 2048))
        if not (1 <= width <= 16384 and 1 <= height <= 16384) or width * height * 16 > 512 * 1024 * 1024:
            raise RouteBlocked('dimensions', 'Shader UV Bake dimensions exceed the 512 MiB pixel budget')
        return width, height

    def describe(node, context=None):
        """Choose IMAGE or UV semantics and return only primitive values."""
        if node is None or getattr(getattr(node, 'id_data', None), 'bl_idname', '') != 'ShaderNodeTree':
            raise RouteBlocked('target', 'Shader Bake requires a Grade in a shader node tree')
        if not _is_grade(node):
            raise RouteBlocked('target', 'Shader Bake is available only for Perceptual Grade and Grade Stack nodes')
        owner_kind, owner, scene, group_path = _owner(node, context)
        images, upstream, graph, state = _dependencies(node, group_path, owner, scene)
        bound_image, bound_kind = _bound_image(node)
        domain = _route_domain(node, state, images, bound_kind)
        _destination_compatible(domain, bound_kind)
        obj = _active_object(context if context is not None else bpy.context, scene)
        material = owner if owner_kind == 'material' else None
        material_slot = None
        uv_map = None
        object_name = ''
        if domain == 'UV':
            if scene.render.engine != 'CYCLES':
                raise RouteBlocked('cycles', 'Shader UV Bake requires the Cycles render engine')
            if material is None:
                raise RouteBlocked('world_uv', 'World Shader Bake supports only direct Image-domain inputs')
            uv_map = _uv(obj)
            material_slot = _material_slot(obj, material)
            object_name = str(obj.name)
        width, height = _size(node, domain, images, bound_image)
        route = dict(domain=domain, width=width, height=height, object_name=object_name,
                     material_name=str(material.name) if material else '', material_slot=material_slot,
                     uv_map=uv_map, images=images, upstream=upstream,
                     intent_upstream=tuple(state.get('intent_upstream', ())),
                     source='constant' if not images else 'image' if domain == 'IMAGE' else 'native')
        target = dict(operation='shader_bake', schema=SCHEMA, scene=str(scene.name),
                      frame=int(scene.frame_current), frame_subframe=float(scene.frame_subframe),
                      engine=str(scene.render.engine), owner_kind=str(owner_kind), owner_name=str(owner.name),
                      group_path=tuple(str(item) for item in group_path), node_name=str(node.name),
                      socket_name='Color', shader_bake=route)
        active_object = _id_key(obj) if domain == 'UV' else None
        mesh_state = None
        if domain == 'UV' and obj is not None:
            if obj.name not in scene.objects:
                raise RouteBlocked('mesh_scene', 'Shader UV Bake active mesh is not in the selected scene')
            if obj.hide_render:
                raise RouteBlocked('mesh_hidden', 'Shader UV Bake active mesh is disabled for rendering')
            mesh_state = (_id_key(obj.data),
                          tuple(tuple(float(value) for value in vertex.co) for vertex in obj.data.vertices),
                          tuple((int(poly.loop_start), int(poly.loop_total), int(poly.material_index))
                                for poly in obj.data.polygons),
                          tuple((int(loop.vertex_index),) for loop in obj.data.loops),
                          tuple((str(layer.name), tuple(tuple(float(value) for value in item.uv)
                                                       for item in layer.data)
                                ) for layer in obj.data.uv_layers),
                          tuple(_rna_state(modifier) for modifier in obj.modifiers),
                          tuple(tuple(float(value) for value in row) for row in obj.matrix_world),
                          tuple(_id_key(slot.material) for slot in obj.material_slots))
        scene_state = (int(scene.frame_current), float(scene.frame_subframe), str(scene.render.engine),
                       int(scene.render.resolution_x), int(scene.render.resolution_y),
                       int(scene.render.resolution_percentage),
                       _rna_state(scene.cycles) if hasattr(scene, 'cycles') else ())
        terminal_state = application.component('graph.lut').scope_signature(node)
        terminal_inputs = tuple((socket.identifier, _plain(getattr(socket, 'default_value', None)),
                                 tuple((link.from_node.name, link.from_socket.identifier, bool(link.is_muted))
                                       for link in _links(socket))) for socket in node.inputs
                                if socket.name in {'Color', 'Alpha'})
        signature = repr((target, terminal_state, terminal_inputs, graph, scene_state,
                          active_object, mesh_state))
        target['identity'] = ('shader-bake-v1', hashlib.sha256(signature.encode('utf8')).hexdigest())
        target['graph'] = graph
        target['terminal_inputs'] = terminal_inputs
        return target

    def preview(node, context=None):
        """Return flat route controls and reason codes without source fingerprinting."""
        if node is None or getattr(getattr(node, 'id_data', None), 'bl_idname', '') != 'ShaderNodeTree':
            raise RouteBlocked('target', 'Shader Bake requires a Grade in a shader node tree')
        if not _is_grade(node):
            raise RouteBlocked('target', 'Shader Bake is available only for Perceptual Grade and Grade Stack nodes')
        owner_kind, owner, _scene, group_path = _owner(node, context)
        _images, _upstream, _graph, state = _dependencies(node, group_path, owner, _scene, fingerprint=False)
        bound_image, bound_kind = _bound_image(node)
        domain = _route_domain(node, state, _images, bound_kind)
        reasons = set(state['reasons'])
        if len(_images) > 1:
            reasons.add('multiple_images')
        if bound_kind == 'SHADER_UV':
            reasons.add('existing_uv_binding')
        if domain == 'UV' and bound_image is not None and bound_kind != 'EMPTY':
            width, height = int(bound_image.get('_pg_bake_width', 0)), int(bound_image.get('_pg_bake_height', 0))
        else:
            width = int(node.get('_pg_bake_width', 2048))
            height = int(node.get('_pg_bake_height', 2048))
        return dict(domain=domain, show_dimensions=domain == 'UV',
                    width=width if domain == 'UV' else None,
                    height=height if domain == 'UV' else None,
                    reasons=tuple(reason for reason in ('existing_uv_binding', 'linked_coordinates',
                                  'multiple_images', 'surface_dependency') if reason in reasons),
                    source='constant' if not _images else 'image' if domain == 'IMAGE' else 'native')

    def observe(node, context=None):
        """Observe eligibility without pixels, filesystem access or UV geometry work."""
        route = None
        try:
            route = preview(node, context)
            owner_kind, owner, scene, group_path = _owner(node, context)
            images, _upstream, _graph, state = _dependencies(
                node, group_path, owner, scene, fingerprint=False)
            bound_image, bound_kind = _bound_image(node)
            _destination_compatible(route['domain'], bound_kind)
            _size(node, route['domain'], images, bound_image, passive=True)
            if route['domain'] == 'UV':
                if scene.render.engine != 'CYCLES':
                    raise RouteBlocked('cycles', 'Shader UV Bake requires the Cycles render engine')
                if owner_kind != 'material':
                    raise RouteBlocked('world_uv', 'World Shader Bake supports only direct Image-domain inputs')
                obj = _active_object(context or bpy.context, scene)
                _uv(obj, inspect_content=False)
                _material_slot(obj, owner)
                if obj.name not in scene.objects:
                    raise RouteBlocked('mesh_scene', 'Shader UV Bake active mesh is not in the selected scene')
                if obj.hide_render:
                    raise RouteBlocked('mesh_hidden', 'Shader UV Bake active mesh is disabled for rendering')
            return dict(state='ready', code='', route=route)
        except RouteBlocked as error:
            return dict(state='blocked', code=error.code, route=route)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return dict(state='blocked', code='unavailable', route=route)

    def observation_key(node, context=None):
        """Capture a scalar invalidation key and an exact owner occurrence."""
        owner_kind, owner, scene, group_path = _owner(node, context)
        images, _upstream, graph, _state = _dependencies(
            node, group_path, owner, scene, fingerprint=False)
        obj = _active_object(context or bpy.context, scene)
        frozen = dict(scene=str(scene.name), owner_kind=owner_kind, owner_name=str(owner.name),
                      group_path=tuple(group_path), node_name=str(node.name),
                      shader_bake=dict(object_name=str(obj.name) if obj is not None else ''))
        source_values = []
        for item in images:
            image = bpy.data.images.get(item['image_name'])
            if image is None:
                raise RouteBlocked('source_unbound', 'Shader Bake source Image is unavailable')
            packed = image.packed_file
            source_values.append((_id_key(image), str(image.source), str(image.filepath),
                bool(image.is_dirty), 0 if packed is None else int(packed.size),
                str(image.colorspace_settings.name), str(image.alpha_mode),
                image.get('_pg_bake_operation'), image.get('_pg_bake_width'), image.get('_pg_bake_height')))
        mesh = getattr(obj, 'data', None) if obj is not None and obj.type == 'MESH' else None
        object_state = None if obj is None else (_id_key(obj), _id_key(mesh),
            bool(obj.hide_render), bool(getattr(mesh, 'is_updated_data', False)),
            len(mesh.vertices) if mesh else 0, len(mesh.loops) if mesh else 0,
            tuple((str(layer.name), len(layer.data), bool(layer.active_render)) for layer in mesh.uv_layers) if mesh else (),
            tuple(_id_key(slot.material) for slot in obj.material_slots),
            tuple(tuple(float(value) for value in row) for row in obj.matrix_world),
            tuple(_rna_state(modifier) for modifier in obj.modifiers))
        destination = bake.baked_image(node)
        destination_state = None if destination is None else (_id_key(destination),
            destination.get('_pg_bake_kind'), destination.get('_pg_bake_operation'),
            bool(destination.is_dirty), destination.get('_pg_bake_width'), destination.get('_pg_bake_height'))
        terminal = tuple((socket.identifier, _plain(getattr(socket, 'default_value', None)),
                          tuple((link.from_node.name, link.from_socket.identifier) for link in _links(socket)))
                         for socket in node.inputs if socket.name in {'Color', 'Alpha'})
        key = (_id_key(scene), int(scene.frame_current), float(scene.frame_subframe),
               str(scene.render.engine), _id_key(owner), tuple(group_path), graph, tuple(source_values),
               object_state, destination_state, terminal, _parameter_observation(node),
               int(node.get('_pg_bake_width', 2048)), int(node.get('_pg_bake_height', 2048)))
        return key, frozen

    def resolve_context(context):
        return _resolve(context, context['group_path'], context['node_name'])

    def _resolve(target, group_path, node_name):
        owner = (bpy.data.materials if target['owner_kind'] == 'material' else bpy.data.worlds).get(
            target['owner_name'])
        if owner is None or owner.node_tree is None:
            raise ReferenceError('Shader Bake owner is unavailable')
        tree = owner.node_tree
        for name in group_path:
            instance = tree.nodes.get(name)
            if instance is None or instance.node_tree is None:
                raise ReferenceError('Shader Bake node-group occurrence changed')
            tree = instance.node_tree
        node = tree.nodes.get(node_name)
        if node is None:
            raise ReferenceError('Shader Bake node changed')
        return node

    def grade_dependencies(node, descriptor=None):
        """Resolve only live Grade/Stack outputs consumed by a shader Bake.

        Baked Color/Alpha are absent because they consume the stored Image.
        Adopted live outputs are observed even though their native resources
        do not belong to managed Grade preparation.
        """
        descriptor = describe(node) if descriptor is None else descriptor
        route = descriptor.get('shader_bake', {})
        values = [node]
        seen = {_pointer(node)}
        for item in route.get('intent_upstream', route.get('upstream', ())):
            value = _resolve(descriptor, item['group_path'], item['node_name'])
            pointer = _pointer(value)
            if pointer and pointer not in seen:
                seen.add(pointer)
                values.append(value)
        return tuple(values)

    def prepare(node, descriptor=None):
        """Make managed upstream Grade/Stack outputs final before freezing."""
        descriptor = describe(node) if descriptor is None else descriptor
        route = descriptor['shader_bake']
        if descriptor.get('identity') != describe(node, descriptor)['identity']:
            raise RuntimeError('Shader Bake inputs changed before preparation')
        for item in route['upstream']:
            upstream = _resolve(descriptor, item['group_path'], item['node_name'])
            if not _is_grade(upstream) or lut.is_adopted(upstream):
                continue
            if not nodes.resources_ready(upstream):
                raise RouteBlocked('upstream', 'An upstream Grade output is not ready')
            if stacks.is_stack(upstream):
                stacks.refresh(upstream)
            else:
                nodes.refresh_node(upstream)
            model = nodes.bake_model(upstream)
            lut.ensure(upstream, model, force=True)
            lut.validate_output(upstream, model, final=True)
        return descriptor

    def freeze(node, descriptor=None):
        descriptor = prepare(node, descriptor)
        frozen = deepcopy(_plain(descriptor))
        frozen['frozen'] = True
        return frozen

    return Component({'__name__': __name__, 'describe': describe, 'preview': preview,
                      'prepare': prepare, 'freeze': freeze, 'observe': observe,
                      'grade_dependencies': grade_dependencies,
                      'observation_key': observation_key, 'resolve_context': resolve_context,
                      'RouteBlocked': RouteBlocked})
