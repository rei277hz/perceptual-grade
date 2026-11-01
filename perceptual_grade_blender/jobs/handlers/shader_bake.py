"""Frozen Shader input rasters, evaluated by Cycles in an isolated helper.

Image-domain bakes use a source-resolution UV plane. UV-domain bakes use the
frozen object's selected material slot and active UV map. Only the selected
terminal Grade/Stack is analytical; upstream native nodes remain native.
"""
from pathlib import Path

import numpy as np

from . import native_capture as native


MAX_PIXEL_BYTES = 512 * 1024 * 1024


def _pixels(image):
    width, height = (int(value) for value in image.size)
    if min(width, height) < 1 or width * height * 16 > MAX_PIXEL_BYTES:
        raise ValueError('Shader bake Image has an invalid or oversized pixel domain')
    values = np.empty(width * height * image.channels, np.float32)
    image.pixels.foreach_get(values)
    values = values.reshape(height, width, image.channels)
    if image.channels in (1, 2):
        alpha = values[..., 1:2] if image.channels == 2 else np.ones((height, width, 1), np.float32)
        values = np.concatenate((np.repeat(values[..., :1], 3, axis=-1), alpha), axis=-1)
    elif image.channels == 3:
        values = np.concatenate((values, np.ones((height, width, 1), np.float32)), axis=-1)
    if values.shape[-1] != 4:
        raise ValueError('Shader bake Image does not contain RGBA pixels')
    return np.ascontiguousarray(values)


def _image(name, width, height, pixels=None):
    import bpy
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.alpha_mode = 'CHANNEL_PACKED'
    if hasattr(image, 'use_half_precision'):
        image.use_half_precision = False
    if pixels is not None:
        image.pixels.foreach_set(np.asarray(pixels, dtype=np.float32).reshape(-1))
        image.update()
    return image


def _tree(root, path):
    tree = root
    for name in path:
        node = tree.nodes.get(name)
        if node is None or getattr(node, 'node_tree', None) is None:
            raise ValueError('Shader bake group occurrence is absent: ' + str(name))
        tree = node.node_tree
    return tree


def _private_paths(root, paths):
    """Detach group instances before changing source or terminal sockets."""
    copied = set()
    for path in paths:
        tree = root
        prefix = ()
        for name in path:
            prefix = (*prefix, name)
            group = tree.nodes.get(name)
            if group is None or getattr(group, 'node_tree', None) is None:
                raise ValueError('Shader bake group occurrence is absent: ' + str(name))
            if prefix not in copied:
                group.node_tree = group.node_tree.copy()
                copied.add(prefix)
            tree = group.node_tree


def _input(tree, socket):
    links = tuple(link for link in socket.links if link.is_valid and not link.is_muted)
    if len(links) > 1:
        raise ValueError('Shader bake input has more than one source')
    if links:
        if links[0].from_socket.type == 'SHADER':
            raise ValueError('Shader closures cannot be baked as Color or Alpha')
        return links[0].from_socket
    node = tree.nodes.new('ShaderNodeRGB' if socket.type == 'RGBA' else 'ShaderNodeValue')
    node.outputs[0].default_value = socket.default_value
    return node.outputs[0]


def _tap(root, path, node_name):
    tree = _tree(root, path)
    node = tree.nodes.get(node_name)
    if node is None or 'Color' not in node.inputs or 'Alpha' not in node.inputs:
        raise ValueError('Shader bake target does not expose Color and Alpha inputs')
    color = _input(tree, node.inputs['Color'])
    alpha = _input(tree, node.inputs['Alpha'])
    # Preserve Blender's conversion into the actual terminal input socket.
    # In particular, Color->Alpha uses luminance and Vector->Color stays RGB.
    for index in range(len(path), 0, -1):
        parent = _tree(root, path[:index - 1])
        group = parent.nodes[path[index - 1]]
        endpoint = next((item for item in tree.nodes if item.bl_idname == 'NodeGroupOutput'
                         and item.is_active_output), None)
        if endpoint is None:
            endpoint = tree.nodes.new('NodeGroupOutput')
            endpoint.is_active_output = True
        exposed = []
        for name, kind, source in (('Color', 'NodeSocketColor', color),
                                   ('Alpha', 'NodeSocketFloat', alpha)):
            field = tree.interface.new_socket(name='PG Bake ' + name, in_out='OUTPUT', socket_type=kind)
            tree.links.new(source, native._socket(endpoint.inputs, field.identifier))
            exposed.append(native._socket(group.outputs, field.identifier))
        color, alpha = exposed
        tree = parent
    # A root Alpha link may be a Color or Vector output. Force the scalar
    # conversion before feeding its three equal emission channels.
    convert = root.nodes.new('ShaderNodeMath')
    convert.operation = 'ADD'
    convert.inputs[1].default_value = 0.
    root.links.new(alpha, convert.inputs[0])
    return color, convert.outputs[0]


def _isolate_alpha(tree, texture, alpha_linked, *, image_domain):
    """Keep the frozen native Image interpretation live in every capture pass.

    Cycles changes its Color result depending on whether Alpha is live, and
    for byte textures unassociates *before* decoding sRGB. Post-Color division
    cannot reproduce that rule. A neutral native Scale retains the original
    Image Texture Alpha dependency: abs(alpha)>-1 is exactly one for every
    finite alpha. Blender 5.2's constant folder keeps this dynamic comparison,
    so the actual Image node performs all color/alpha conversion itself.
    """
    image = texture.image
    if image is None or image.source not in {'FILE', 'GENERATED'} or image.type not in {'IMAGE', 'UV_TEST'}:
        raise ValueError('Shader Bake supports ordinary still Images; use the automatic LUT for UDIM or animation')
    if image.use_multiview:
        raise ValueError('Shader Bake does not support multiview Images')
    if image_domain:
        for link in tuple(texture.inputs['Vector'].links):
            tree.links.remove(link)
        texture.projection = 'FLAT'
        texture.interpolation = 'Closest'
        texture.extension = 'EXTEND'
        texture.texture_mapping.translation = (0., 0., 0.)
        texture.texture_mapping.rotation = (0., 0., 0.)
        texture.texture_mapping.scale = (1., 1., 1.)
        texture.texture_mapping.use_min = texture.texture_mapping.use_max = False
        texture.texture_mapping.vector_type = 'POINT'
        texture.texture_mapping.mapping_x = 'X'
        texture.texture_mapping.mapping_y = 'Y'
        texture.texture_mapping.mapping_z = 'Z'
        texture.texture_mapping.mapping = 'FLAT'
    color_links = tuple(texture.outputs['Color'].links)
    if alpha_linked and color_links:
        absolute = tree.nodes.new('ShaderNodeMath')
        absolute.operation = 'ABSOLUTE'
        tree.links.new(texture.outputs['Alpha'], absolute.inputs[0])
        positive = tree.nodes.new('ShaderNodeMath')
        positive.operation = 'GREATER_THAN'
        positive.inputs[1].default_value = -1.
        tree.links.new(absolute.outputs[0], positive.inputs[0])
        scale = tree.nodes.new('ShaderNodeVectorMath')
        scale.operation = 'SCALE'
        tree.links.new(texture.outputs['Color'], scale.inputs[0])
        tree.links.new(positive.outputs[0], scale.inputs['Scale'])
        for link in color_links:
            destination = link.to_socket
            tree.links.remove(link)
            tree.links.new(scale.outputs['Vector'], destination)
    return texture


def _material(root, color, alpha):
    """Use the native owner graph; a World tree can also be a private group."""
    import bpy
    material = bpy.data.materials.new('PG Bake Input')
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    endpoint = next((item for item in root.nodes if item.bl_idname == 'NodeGroupOutput'
                     and item.is_active_output), None)
    if endpoint is None:
        endpoint = root.nodes.new('NodeGroupOutput')
        endpoint.is_active_output = True
    fields = []
    for name, kind, source in (('Color', 'NodeSocketColor', color), ('Alpha', 'NodeSocketFloat', alpha)):
        field = root.interface.new_socket(name='PG Bake ' + name, in_out='OUTPUT', socket_type=kind)
        root.links.new(source, native._socket(endpoint.inputs, field.identifier))
        fields.append(field.identifier)
    group = tree.nodes.new('ShaderNodeGroup')
    group.node_tree = root
    emission = tree.nodes.new('ShaderNodeEmission')
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.is_active_output = True
    tree.links.new(emission.outputs[0], output.inputs['Surface'])
    return material, emission, tuple(native._socket(group.outputs, name) for name in fields)


def _plane(scene):
    import bpy
    mesh = bpy.data.meshes.new('PG Bake Image Plane')
    mesh.from_pydata(((-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.)), (), ((0, 1, 2, 3),))
    uv = mesh.uv_layers.new(name='PG Bake UV')
    for loop, value in zip(uv.data, ((0., 0.), (1., 0.), (1., 1.), (0., 1.))):
        loop.uv = value
    mesh.uv_layers.active = uv
    uv.active_render = True
    obj = bpy.data.objects.new('PG Bake Image Plane', mesh)
    scene.collection.objects.link(obj)
    return obj


def _select(scene, obj, uv_map):
    import bpy
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    layer = bpy.context.view_layer
    # Bake the frozen selected material itself. A render-layer override replaces
    # that material before Cycles evaluates its active Image Texture.
    layer.material_override = None
    for item in layer.objects:
        item.select_set(False)
    obj.hide_render = False
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.select_set(True)
    layer.objects.active = obj
    uv = obj.data.uv_layers.get(uv_map)
    if uv is None:
        raise ValueError('Shader bake UV map is absent: ' + str(uv_map))
    obj.data.uv_layers.active = uv
    uv.active_render = True
    layer.update()


def _sink(material, image):
    tree = material.node_tree
    node = tree.nodes.new('ShaderNodeTexImage')
    node.image = image
    for item in tree.nodes:
        item.select = item == node
    tree.nodes.active = node
    return node


def _bake(scene, obj, material, emission, output, image, context, margin):
    import bpy
    context.checkpoint()
    tree = material.node_tree
    for link in tuple(emission.inputs['Color'].links):
        tree.links.remove(link)
    tree.links.new(output, emission.inputs['Color'])
    # Blender's bake operator reads the actual material's active image node.
    sink = next((node for node in tree.nodes if node.bl_idname == 'ShaderNodeTexImage' and node.image == image), None)
    if sink is None:
        sink = _sink(material, image)
    tree.nodes.active = sink
    scene.render.bake.margin = margin
    scene.render.bake.margin_type = 'EXTEND'
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.target = 'IMAGE_TEXTURES'
    result = bpy.ops.object.bake(type='EMIT', margin=margin, use_clear=True, use_selected_to_active=False)
    context.checkpoint()
    if 'FINISHED' not in result:
        raise RuntimeError('Native Shader Bake did not finish')
    return _pixels(image)


def evaluate(scene, root, target, descriptor, context):
    """Evaluate one already loaded, privately owned frozen native scene."""
    import bpy
    domain = descriptor.get('domain')
    if domain not in {'IMAGE', 'UV'}:
        raise ValueError('Unknown Shader Bake domain')
    width, height = int(descriptor['width']), int(descriptor['height'])
    if min(width, height) < 1 or max(width, height) > 32768 or width * height * 16 > MAX_PIXEL_BYTES:
        raise ValueError('Shader Bake dimensions exceed the pixel budget')
    if domain == 'UV' and scene.render.engine != 'CYCLES':
        raise ValueError('UV Shader Bake requires Cycles')
    context.checkpoint()
    occurrences = tuple(descriptor.get('images', ()))
    path = tuple(target.get('group_path', ()))
    _private_paths(root, (path, *(tuple(item.get('group_path', ())) for item in occurrences)))
    for item in occurrences:
        tree = _tree(root, item.get('group_path', ()))
        source = tree.nodes.get(item['node_name'])
        if source is None or source.bl_idname != 'ShaderNodeTexImage':
            raise ValueError('Shader Bake Image occurrence is absent')
        _isolate_alpha(tree, source, bool(item.get('alpha_linked')), image_domain=domain == 'IMAGE')
        context.checkpoint()
    color, alpha = _tap(root, path, target['node_name'])
    material, emission, outputs = _material(root, color, alpha)
    image = _image('PG Bake Result', width, height)
    _sink(material, image)
    if domain == 'IMAGE':
        obj = _plane(scene)
        obj.data.materials.append(material)
        uv_map = obj.data.uv_layers.active.name
    else:
        obj = bpy.data.objects.get(descriptor.get('object_name', ''))
        if obj is None or obj.type != 'MESH' or obj.name not in scene.objects:
            raise ValueError('UV Shader Bake requires the frozen active mesh object')
        if any(modifier.type in {'MESH_SEQUENCE_CACHE', 'MESH_CACHE'} or
               getattr(modifier, 'cache_file', None) is not None for modifier in obj.modifiers):
            raise ValueError('Alembic-dependent Shader Bake is unsupported; use the automatic LUT')
        index = int(descriptor.get('material_slot', -1))
        if index < 0 or index >= len(obj.material_slots):
            raise ValueError('UV Shader Bake material slot is absent')
        chosen = obj.material_slots[index].material
        if chosen is None or chosen.name != descriptor.get('material_name', chosen.name):
            raise ValueError('UV Shader Bake material changed after snapshot')
        uv_map = descriptor.get('uv_map', '')
        if not uv_map:
            raise ValueError('UV Shader Bake requires an active UV map')
        # Other slots have their own private sink, leaving uncovered pixels in
        # the selected material's image at zero RGBA.
        scratch = _image('PG Bake Other Materials', width, height)
        for slot in obj.material_slots:
            slot.link = 'OBJECT'
            if slot.slot_index == index:
                slot.material = material
            else:
                other = bpy.data.materials.new('PG Bake Excluded Material')
                other.use_nodes = True
                _sink(other, scratch)
                slot.material = other
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.device = 'CPU'
    scene.render.use_sequencer = False
    scene.compositing_node_group = None
    _select(scene, obj, uv_map)
    rgb = _bake(scene, obj, material, emission, outputs[0], image, context, 16 if domain == 'UV' else 0)
    context.progress(.12)
    alpha_pixels = _bake(scene, obj, material, emission, outputs[1], image, context, 16 if domain == 'UV' else 0)
    result = rgb
    result[..., 3] = alpha_pixels[..., 0]
    # The native bake buffer's A is not a reliable coverage mask: clearing a
    # generated image can leave A=1 outside the island. Capture constant white
    # through the same material and margin instead. It also preserves covered
    # texels whose independently captured Alpha is zero.
    if domain == 'UV':
        white = material.node_tree.nodes.new('ShaderNodeValue')
        white.outputs[0].default_value = 1.
        mask = _bake(scene, obj, material, emission, white.outputs[0], image, context, 16)
        coverage = mask[..., 0] > 0.
    else:
        coverage = np.ones((height, width), dtype=bool)
    if domain == 'UV' and not np.any(coverage):
        raise ValueError('UV Shader Bake selected material has no covered UV pixels')
    context.progress(.2)
    return result, coverage


def handle(inputs, context):
    """Worker operation. Artifacts remain owned by the foreground bake lease."""
    import bpy
    from ....perceptual_grade_core.jobs.codec import thaw_model
    from ....perceptual_grade_core.model import raster
    context.checkpoint()
    target = inputs.get('target', inputs)
    descriptor = inputs.get('shader_bake', target.get('shader_bake'))
    if not isinstance(descriptor, dict):
        raise ValueError('Shader Bake has no frozen domain descriptor')
    directory = Path(inputs['artifact_dir'])
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError('Invalid Shader Bake artifact directory')
    snapshot = inputs['snapshot']
    if isinstance(snapshot, str):
        snapshot = dict(path=snapshot, sha256=inputs['snapshot_sha256'])
    path = native._verified_file(snapshot)
    native._minimal_node_types()
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False, load_ui=False)
    context.checkpoint()
    scene = bpy.data.scenes.get(target['scene'])
    if scene is None:
        raise ValueError('Shader Bake scene is absent from snapshot')
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
        layer_name = inputs.get('view', {}).get('view_layer') or target.get('view_layer')
        if layer_name:
            layer = scene.view_layers.get(layer_name)
            if layer is None:
                raise ValueError('Shader Bake view layer is absent from snapshot')
            bpy.context.window.view_layer = layer
    native._restore_images(inputs.get('images', ()))
    native._restore_clips(inputs.get('clips', ()))
    native._disable_file_outputs()
    owner_kind = target['owner_kind'].lower()
    if owner_kind not in {'material', 'world'} or descriptor['domain'] == 'UV' and owner_kind != 'material':
        raise ValueError('Shader Bake owner is not supported in this domain')
    owner = (bpy.data.materials if owner_kind == 'material' else bpy.data.worlds).get(target['owner_name'])
    if owner is None or owner.node_tree is None:
        raise ValueError('Shader Bake owner is absent from snapshot')
    analytic = inputs.get('analytic', {})
    if analytic.get('version') != raster.SCHEMA or analytic.get('kind') not in {'GRADE', 'STACK'}:
        raise ValueError('Unsupported analytical Shader Bake model')
    pixels, coverage = evaluate(scene, owner.node_tree, target, descriptor, context)
    diagnostics = raster.grade_rgba(pixels, thaw_model(analytic['model']), analytic['rows'],
                                   stack=analytic['kind'] == 'STACK', checkpoint=context.checkpoint,
                                   progress=lambda value: context.progress(.2 + .75 * value))
    if descriptor['domain'] == 'UV':
        pixels[~coverage] = 0.
    context.checkpoint()
    height, width = pixels.shape[:2]
    image = _image('PG Bake Analytic Result', width, height, pixels)
    output = directory / 'pg_bake_shader_analytic.exr'
    try:
        native._settings(scene, False)
        image.save_render(str(output), scene=scene)
        context.checkpoint()
        first_image = descriptor.get('images', ())[0] if descriptor.get('images') else {}
        return dict(path=str(output), sha256=native._hash_file(output),
                    width=width, height=height, domain=(width, height),
                    identity=inputs.get('identity'), scene=target['scene'],
                    frame=int(scene.frame_current), frame_subframe=float(scene.frame_subframe),
                    engine='CYCLES', node=target['node_name'], socket='Color',
                    label='Shader ' + descriptor['domain'].lower() + ' bake', backend='CYCLES',
                    evaluation='ANALYTIC', analytic_kind=analytic['kind'],
                    analytic_signature=analytic['signature'], diagnostics=diagnostics,
                    bake_domain=descriptor['domain'], object=descriptor.get('object_name', ''),
                    uv_map=descriptor.get('uv_map', ''),
                    interpolation=first_image.get('interpolation', ''),
                    extension=first_image.get('extension', ''))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        bpy.data.images.remove(image)
