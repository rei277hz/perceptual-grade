"""Evaluate a Color input with Blender in an isolated capture helper.

Only the persisted PG node ABI is registered here.  Product Sessions, timers,
and UI handlers are never enabled.  Blender owns graph evaluation; this
adapter adds output taps and returns immutable scene-linear sample values.
"""
import hashlib
from pathlib import Path
import tempfile
import time
from contextlib import nullcontext

import numpy as np


def _checkpoint(context):
    context.checkpoint()


def _verified_file(record):
    path = Path(record['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Invalid native capture artifact')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError('Native capture artifact changed after snapshot')
    return path


def _minimal_node_types():
    """Persisted native group kernels do not require the extension lifecycle."""
    import bpy
    for base, identifier in (
        ('NodeTree', 'PGGradeData'),
        ('ShaderNodeCustomGroup', 'PGShaderGrade'),
        ('CompositorNodeCustomGroup', 'PGCompositorGrade'),
        ('ShaderNodeCustomGroup', 'PGShaderStack'),
        ('CompositorNodeCustomGroup', 'PGCompositorStack'),
        # Bake nodes are ordinary persisted custom groups.  The isolated helper
        # only needs the ABI class so Blender can load the frozen scene; the
        # foreground component owns the actual node implementation.
        ('CompositorNodeCustomGroup', 'PGCompositorBake'),
        ('ShaderNodeCustomGroup', 'PGShaderBake'),
    ):
        # Node subclasses are queried through Node, not the custom-group base.
        registry = bpy.types.NodeTree if base == 'NodeTree' else bpy.types.Node
        if registry.bl_rna_get_subclass_py(identifier, None) is not None:
            continue
        cls = type(identifier, (getattr(bpy.types, base),), {
            '__module__': __name__, 'bl_idname': identifier,
            'bl_label': identifier, 'bl_icon': 'COLOR',
        })
        bpy.utils.register_class(cls)


def _socket(sockets, identifier):
    result = next((socket for socket in sockets if socket.identifier == identifier), None)
    if result is None:
        result = sockets.get(identifier)
    if result is None:
        raise ValueError('Native capture socket no longer exists: ' + str(identifier))
    return result


def _source(root, group_path, node_name, socket_name, token):
    """Expose one exact group occurrence as a Color output on its owner tree.

    Copy each group along the selected instance path before adding interface
    outputs, so a reused group elsewhere cannot emit this capture's AOV.
    Blender supplies any implicit conversion into the Color tap.
    """
    tree = root
    parents = []
    for name in group_path:
        group = tree.nodes.get(name)
        if group is None or getattr(group, 'node_tree', None) is None:
            raise ValueError('Native capture group instance no longer exists: ' + str(name))
        group.node_tree = group.node_tree.copy()
        parents.append((tree, group))
        tree = group.node_tree
    target = tree.nodes.get(node_name)
    if target is None:
        raise ValueError('Native capture node no longer exists: ' + str(node_name))
    incoming = _socket(target.inputs, socket_name)
    if incoming.type != 'RGBA':
        raise ValueError('Native capture requires a Color input')
    links = tuple(link for link in incoming.links if not link.is_muted and link.is_valid)
    if links:
        output = links[0].from_socket
        if output.type == 'SHADER':
            raise ValueError('Shader closures cannot be sampled as a Color input')
    else:
        node = tree.nodes.new('ShaderNodeRGB' if tree.bl_idname == 'ShaderNodeTree' else 'CompositorNodeRGB')
        node.outputs[0].default_value = tuple(incoming.default_value)
        output = node.outputs[0]
    for parent, group in reversed(parents):
        field = tree.interface.new_socket(name='PG Scope ' + token, in_out='OUTPUT', socket_type='NodeSocketColor')
        endpoint = next((node for node in tree.nodes if node.bl_idname == 'NodeGroupOutput'
                         and node.is_active_output), None)
        if endpoint is None:
            endpoint = tree.nodes.new('NodeGroupOutput')
            endpoint.is_active_output = True
        tree.links.new(output, _socket(endpoint.inputs, field.identifier))
        output = _socket(group.outputs, field.identifier)
        tree = parent
    return output


def _restore_images(records):
    """Rebind captured volatile buffers only in this helper's Main database."""
    import bpy
    restored = {}
    for record in records:
        original = bpy.data.images.get(record['name'])
        if original is None:
            raise ValueError('Captured image is absent from snapshot: ' + record['name'])
        path = _verified_file(record)
        if record.get('source') == 'file':
            for member in record.get('files', ()):
                _verified_file(member)
            original.filepath = record.get('filepath', str(path))
            original.reload()
            restored[record['name']] = original
            continue
        pixels = np.load(path, allow_pickle=False)
        if pixels.ndim != 3 or pixels.shape[2] not in (1, 2, 3, 4) or pixels.dtype.kind != 'f':
            raise ValueError('Invalid captured image buffer')
        height, width, channels = pixels.shape
        if channels in (1, 2):
            rgb = np.repeat(pixels[..., :1], 3, axis=-1)
            alpha = pixels[..., 1:2] if channels == 2 else np.ones((height, width, 1), np.float32)
            pixels = np.concatenate((rgb, alpha), axis=-1)
        elif channels == 3:
            pixels = np.concatenate((pixels, np.ones((height, width, 1), np.float32)), axis=-1)
        replacement = bpy.data.images.new('PG Capture ' + record['name'], width=width,
                                          height=height, alpha=True,
                                          float_buffer=record.get('is_float', original.is_float))
        # Byte pixels are encoded/straight, while float pixels already follow
        # Blender's scene-linear buffer convention. Preserve the original
        # buffer type and interpretation so native evaluation applies exactly
        # the same color and alpha conversion as the foreground image.
        replacement.colorspace_settings.name = record.get(
            'colorspace', original.colorspace_settings.name)
        replacement.alpha_mode = record.get('alpha_mode', original.alpha_mode)
        replacement.pixels.foreach_set(np.asarray(pixels, dtype=np.float32).ravel())
        replacement.update()
        original.user_remap(replacement)
        restored[record['name']] = replacement
    return restored


def _companion(inputs, restored, context):
    """Decode a raw companion from the same frozen scene as its native input."""
    if '_image_source' not in inputs and '_constant_source' not in inputs:
        return None
    import bpy
    from dataclasses import replace
    from ...integration.components import Component
    from ...scopes import images, exr
    from ....perceptual_grade_core.scopes.samples import ordinary_layout, immutable

    layout = _sample_layout(inputs.get('sample_grid'))
    context.checkpoint()
    if '_image_source' in inputs:
        value = dict(inputs['_image_source'])
        value['frame_settings'] = images.FrameSettings(**value['frame_settings'])
        descriptor = images.SourceDescriptor(**value)
        image = restored.get(descriptor.name) or bpy.data.images.get(descriptor.name)
        if image is None:
            raise ValueError('Frozen companion Image is absent from the snapshot')
        class PrivateApplication:
            def component(self, name):
                if name == 'graph.output_guard':return Component({'native_operation': nullcontext})
                if name == 'scopes.exr':return exr.create(None)
                raise ValueError('Private companion decoder requested an application service')
        # Volatile buffers are restored as private generated Images; their
        # original interpretation and frozen animation selection are retained.
        local = replace(descriptor, lifetime=(0, int(image.session_uid)),
            source=str(image.source), image_type=str(image.type),
            filepath=bpy.path.abspath(image.filepath, library=image.library) if image.filepath else '')
        reader = images.create(PrivateApplication(), background_movies=True)
        sample = reader.read(local, layout=layout, force=True)
        rgba, layout, label = sample.rgba, sample.layout, sample.label
        domain = tuple(int(v) for v in image.size)
    else:
        values = np.asarray(inputs['_constant_source'], dtype=np.float32)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError('Invalid frozen constant companion')
        layout = layout or ordinary_layout(256, 1)
        rgba = immutable(np.broadcast_to(values, (*layout.mass.shape, 4)).copy())
        label, domain = 'Constant input', (1, 1)
    context.checkpoint()
    return dict(rgba=rgba, identity=inputs.get('identity'), domain=domain,
        label=label, backend='source', prepared_provenance=(),
        layout={name:getattr(layout,name) for name in ('mass','x','uv','tiles','ids','key','tiled')})


def _restore_clips(records):
    import bpy
    for record in records:
        original = bpy.data.movieclips.get(record['name'])
        if original is None:
            raise ValueError('Captured movie clip is absent from snapshot: ' + record['name'])
        path = _verified_file(record)
        for member in record.get('files', ()):
            _verified_file(member)
        original.filepath = record.get('filepath', str(path))
        # MovieClip has no Image-style reload method in Blender 5.2. Updating
        # its filepath invalidates the native clip source before evaluation.


def _disable_file_outputs():
    import bpy
    # Scope evaluation cannot write user File Output paths, even in a copied
    # scene: native render evaluates active output nodes beyond our tap.
    for tree in bpy.data.node_groups:
        if tree.bl_idname == 'CompositorNodeTree':
            for node in list(tree.nodes):
                if node.bl_idname == 'CompositorNodeOutputFile':
                    tree.nodes.remove(node)


def _orthographic_projection(values, clip_start):
    """Recover a parallel camera's depth interval from a viewport matrix."""
    projection = np.asarray(values, dtype=np.float64)
    error = 'Native scope camera cannot reproduce the selected viewport projection'
    if projection.shape != (4, 4) or not np.isfinite(projection).all():
        raise ValueError(error)
    if projection[3, 3] == 0:
        raise ValueError(error)
    projection = projection / projection[3, 3]
    allowed = {(0, 0), (1, 1), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)}
    if (not np.isfinite(projection).all() or projection[0, 0] <= 0
            or projection[1, 1] <= 0
            or any(abs(projection[row, column]) > 1e-7
                   for row in range(4) for column in range(4)
                   if (row, column) not in allowed)):
        raise ValueError(error)
    scale, offset = projection[2, 2], projection[2, 3]
    if scale >= 0 or not np.isfinite(clip_start):
        raise ValueError(error)
    near_z, far_z = (-1.0 - offset) / scale, (1.0 - offset) / scale
    near = max(.001, float(clip_start))
    far = near + near_z - far_z
    # Viewport orthographic clipping can straddle its origin. A render
    # camera only sees negative local Z, so move its origin behind the
    # entire frozen interval without changing its parallel viewing rays.
    displacement = near_z + near
    if not np.isfinite((near, far, displacement)).all() or not 0 < near < far:
        raise ValueError(error)
    return projection, near, far, displacement


def _camera(scene, view):
    import bpy
    from mathutils import Matrix
    if 'view_matrix' not in view or 'projection_matrix' not in view:
        camera_name = view.get('camera')
        if camera_name:
            scene.camera = bpy.data.objects.get(camera_name)
        if scene.camera is None:
            raise ValueError('Shader scopes need a 3D viewport or an active scene camera')
        return
    matrix = Matrix(view['view_matrix'])
    projection = Matrix(view['projection_matrix'])
    matrix_scale = max(abs(value) for row in projection for value in row)
    if not np.isfinite(matrix_scale) or matrix_scale == 0:
        raise ValueError('Native scope camera cannot reproduce the selected viewport projection')
    orthographic = abs(projection[3][3]) > matrix_scale * 1e-6
    if orthographic:
        projection, near, far, displacement = _orthographic_projection(
            projection, float(view.get('clip_start', .01)))
        kind = 'ORTHO'
    else:
        # Keep the existing perspective path and viewport clip settings.
        near = max(0.001, float(view.get('clip_start', .01)))
        far = max(near + .001, float(view.get('clip_end', 1000.)))
        displacement = 0.0
        kind = 'PERSP'
    camera_data = bpy.data.cameras.new('PG Scope Camera')
    camera = bpy.data.objects.new('PG Scope Camera', camera_data)
    scene.collection.objects.link(camera)
    depth_origin = Matrix.Translation((0.0, 0.0, displacement))
    camera.matrix_world = matrix.inverted() @ depth_origin
    scene.camera = camera
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 36.0
    camera_data.clip_start = near
    camera_data.clip_end = far
    camera_data.type = kind
    if kind == 'PERSP':
        camera_data.lens = 18.0 * projection[0][0]
        camera_data.shift_x = projection[0][2] / 2.0
        camera_data.shift_y = projection[1][2] * projection[0][0] / (2.0 * projection[1][1])
    else:
        camera_data.ortho_scale = 2.0 / projection[0][0]
        camera_data.shift_x = -projection[0][3] / 2.0
        camera_data.shift_y = -projection[1][3] * projection[0][0] / (2.0 * projection[1][1])
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    # Matching XY alone cannot detect geometry behind an orthographic camera.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    actual = camera.calc_matrix_camera(depsgraph, x=scene.render.resolution_x,
                                       y=scene.render.resolution_y,
                                       scale_x=scene.render.pixel_aspect_x,
                                       scale_y=scene.render.pixel_aspect_y)
    expected = np.asarray(projection)
    if orthographic:
        effective = np.asarray(actual @ depth_origin.inverted())
    else:
        effective, expected = np.asarray(actual)[:2], expected[:2]
    if not np.allclose(effective, expected, rtol=2e-5, atol=2e-5):
        raise ValueError('Native scope camera cannot reproduce the selected viewport projection')
    visible = view.get('visible_objects')
    if visible is not None:
        names = set(visible)
        # The foreground visibility set comes from the active viewport. Clear
        # collection render gates first so a render-only collection flag cannot
        # reintroduce a hidden object or suppress one the viewport displayed.
        for collection in bpy.data.collections:
            collection.hide_render = False
        for obj in scene.objects:
            if obj != camera:
                obj.hide_render = obj.name not in names


def _aov(view_layer, tree, name, output=None):
    entry = view_layer.aovs.add()
    entry.name = name
    entry.type = 'COLOR'
    tap = tree.nodes.new('ShaderNodeOutputAOV')
    tap.aov_name = name
    if output is None:
        tap.inputs['Color'].default_value = (1., 1., 1., 1.)
    else:
        tree.links.new(output, tap.inputs['Color'])


def _shader(scene, view_layer, target, view, token):
    import bpy
    if scene.render.engine not in ('CYCLES', 'BLENDER_EEVEE'):
        raise ValueError('The scene render engine does not provide native shader scope AOVs: ' + scene.render.engine)
    kind = target['owner_kind'].lower()
    owner = (bpy.data.materials if kind == 'material' else bpy.data.worlds).get(target['owner_name'])
    if owner is None or owner.node_tree is None:
        raise ValueError('Native scope shader owner is unavailable: ' + target['owner_name'])
    if kind == 'world' and scene.world != owner:
        raise ValueError('The selected World is not used by this scene')
    output = _source(owner.node_tree, target.get('group_path', ()), target['node_name'],
                     target.get('socket_name', 'Color'), token)
    color_name, mask_name = 'PGScopeColor_' + token, 'PGScopeMask_' + token
    _aov(view_layer, owner.node_tree, color_name, output)
    _aov(view_layer, owner.node_tree, mask_name)
    scene.compositing_node_group = None
    scene.render.use_sequencer = False
    _camera(scene, view)
    return color_name, mask_name


def _compositor(scene, target, token, directory):
    tree = scene.compositing_node_group
    if tree is None:
        raise ValueError('Scene has no compositor graph')
    output = _source(tree, target.get('group_path', ()), target['node_name'],
                     target.get('socket_name', 'Color'), token)
    # Remove downstream outputs so disconnected scene branches do not trigger
    # unrelated rendering. A root tap also supplies the scene-sized fallback
    # for a constant signal, which has no image domain for File Output to save.
    for node in list(tree.nodes):
        if node.bl_idname in ('NodeGroupOutput', 'CompositorNodeViewer'):
            tree.nodes.remove(node)
    field = next((item for item in tree.interface.items_tree if item.item_type == 'SOCKET'
                  and item.in_out == 'OUTPUT' and item.socket_type == 'NodeSocketColor'), None)
    if field is None:
        field = tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    root_output = tree.nodes.new('NodeGroupOutput')
    root_output.is_active_output = True
    tree.links.new(output, _socket(root_output.inputs, field.identifier))
    # A root Group Output is bounded to the scene render canvas. A compositor
    # File Output receives the selected upstream image at its native domain,
    # including image inputs larger or differently sized than that canvas.
    endpoint = tree.nodes.new('CompositorNodeOutputFile')
    endpoint.name = 'PG Scope Capture'
    endpoint.directory = str(directory)
    endpoint.file_name = 'pg_scope_capture_' + token + '_'
    item = endpoint.file_output_items.new('RGBA', 'scope')
    endpoint.format.media_type = 'IMAGE'
    endpoint.format.file_format = 'OPEN_EXR'
    endpoint.format.color_mode = 'RGBA'
    endpoint.format.color_depth = '32'
    endpoint.format.exr_codec = 'ZIPS'
    endpoint.format.color_management = 'OVERRIDE'
    endpoint.format.linear_colorspace_settings.name = 'scene_linear'
    endpoint.save_as_render = False
    tree.links.new(output, endpoint.inputs[item.name])
    scene.render.use_compositing = True
    scene.render.use_sequencer = False


def _bake_compositor(scene, target, token, directory):
    """Capture every pixel at the selected input before analytic grading.

    This is deliberately separate from ``_compositor``: scopes retain their
    bounded sampling contract while a bake keeps every pixel in Blender's
    native output domain.  The output directory is owned by the foreground
    capture lease and therefore must survive this helper's return.
    """
    tree = scene.compositing_node_group
    if tree is None:
        raise ValueError('Scene has no compositor graph')
    output = _source(tree, target.get('group_path', ()), target['node_name'],
                     target.get('socket_name', 'Color'), token)
    for node in list(tree.nodes):
        if node.bl_idname in ('NodeGroupOutput', 'CompositorNodeViewer'):
            tree.nodes.remove(node)
    field = next((item for item in tree.interface.items_tree
                  if item.item_type == 'SOCKET' and item.in_out == 'OUTPUT'
                  and item.socket_type == 'NodeSocketColor'), None)
    if field is None:
        field = tree.interface.new_socket(name='Image', in_out='OUTPUT',
                                          socket_type='NodeSocketColor')
    root_output = tree.nodes.new('NodeGroupOutput')
    root_output.is_active_output = True
    tree.links.new(output, _socket(root_output.inputs, field.identifier))
    endpoint = tree.nodes.new('CompositorNodeOutputFile')
    endpoint.name = 'PG Bake Capture'
    endpoint.directory = str(directory)
    endpoint.file_name = 'pg_bake_' + token + '_'
    item = endpoint.file_output_items.new('RGBA', 'bake')
    endpoint.format.media_type = 'IMAGE'
    endpoint.format.file_format = 'OPEN_EXR'
    endpoint.format.color_mode = 'RGBA'
    endpoint.format.color_depth = '32'
    endpoint.format.exr_codec = 'ZIPS'
    endpoint.format.color_management = 'OVERRIDE'
    endpoint.format.linear_colorspace_settings.name = 'scene_linear'
    endpoint.save_as_render = False
    tree.links.new(output, endpoint.inputs[item.name])
    scene.render.use_compositing = True
    scene.render.use_sequencer = False


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sample_size(width, height, maximum):
    scale = min(1., maximum[0] / width, maximum[1] / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _settings(scene, multilayer):
    settings = scene.render.image_settings
    settings.media_type = 'MULTI_LAYER_IMAGE' if multilayer else 'IMAGE'
    settings.file_format = 'OPEN_EXR_MULTILAYER' if multilayer else 'OPEN_EXR'
    settings.color_mode = 'RGBA'
    settings.color_depth = '32'
    settings.exr_codec = 'ZIPS'
    settings.use_exr_interleave = True
    settings.color_management = 'OVERRIDE'
    settings.linear_colorspace_settings.name = 'scene_linear'


def _sample_layout(grid):
    """Reconstruct the frozen primary grid for a native reference input."""
    if grid is None:return None
    from ....perceptual_grade_core.scopes.samples import SampleLayout, immutable, MAX_SAMPLES
    fields={name:immutable(grid[name]) for name in ('mass','x','uv','tiles','ids')}
    mass,uv=fields['mass'],fields['uv']
    if (mass.ndim!=2 or mass.size>MAX_SAMPLES or min(mass.shape)<1
            or uv.shape!=(*mass.shape,2)
            or any(fields[name].shape!=mass.shape for name in ('x','tiles','ids'))
            or not np.isfinite(mass).all() or np.any(mass<0)
            or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1)):
        raise ValueError('Invalid native scope sample grid')
    key=tuple(grid['key'])
    return SampleLayout(**fields,key=key,tiled=key[0]=='scope-tiles-v1')


def _read_compositor(path, maximum, exact_size=False, layout=None):
    import bpy
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        image.alpha_mode = 'CHANNEL_PACKED'
        width, height = image.size
        if min(width, height) < 1:
            raise ValueError('Compositor scope capture has no pixels')
        count = width * height * image.channels
        pixels = np.empty(count, np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(height, width, image.channels)
        size = maximum if exact_size else _sample_size(width, height, maximum)
        if layout is None:
            xs = np.minimum(width - 1, ((np.arange(size[0]) + .5) * width / size[0]).astype(int))
            ys = np.minimum(height - 1, ((np.arange(size[1]) + .5) * height / size[1]).astype(int))
            sampled = pixels[ys[:, None], xs[None, :]]
        else:
            from ....perceptual_grade_core.scopes.samples import sample_rgba
            sampled=sample_rgba(pixels,layout)
        if image.channels == 3:
            sampled = np.concatenate((sampled, np.ones((*sampled.shape[:2], 1), np.float32)), axis=-1)
        if sampled.shape[2] != 4:
            raise ValueError('Compositor scope capture is not RGBA')
        return np.array(sampled, copy=True), (width, height)
    finally:
        bpy.data.images.remove(image)


def _read_shader(path, layer, color, mask, maximum, exact_size=False, layout=None):
    from ...scopes.exr import create
    reader = create(None)
    with path.open('rb') as stream:
        _, (xmin, ymin, xmax, ymax), _ = reader._header(stream)
    domain = xmax - xmin + 1, ymax - ymin + 1
    size = maximum if exact_size else _sample_size(*domain, maximum)
    rgba = np.array(reader.sample(path, layer, color, size,layout=layout), copy=True)
    coverage = np.clip(reader.sample(path, layer, mask, size,layout=layout)[..., 0], 0., 1.)
    np.divide(rgba[..., :3], coverage[..., None], out=rgba[..., :3],
              where=coverage[..., None] > 1e-8)
    rgba[..., :3][coverage <= 1e-8] = 0.
    rgba[..., 3] = coverage
    return rgba, domain


def handle(inputs, context):
    """Public native-capture operation; only primitive identities cross IPC."""
    import bpy
    measured = bool(inputs.get('_capture_diagnostics'))
    started = time.perf_counter() if measured else None
    timings = {'worker_entered_at': started} if measured else {}
    previous = started

    def mark(phase):
        nonlocal previous
        if measured:
            now = time.perf_counter()
            timings[phase + '_seconds'] = now - previous
            previous = now

    bake_mode = bool(inputs.get('_bake'))
    _checkpoint(context)
    snapshot = inputs['snapshot']
    if isinstance(snapshot, str):
        snapshot = dict(path=snapshot, sha256=inputs['snapshot_sha256'])
    path = _verified_file(snapshot)
    mark('snapshot_verification')
    _minimal_node_types()
    mark('node_type_registration')
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False, load_ui=False)
    mark('blend_load')
    target = inputs.get('target', inputs)
    scene = bpy.data.scenes.get(target['scene'])
    if scene is None:
        raise ValueError('Native capture scene is absent from snapshot')
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    view = inputs.get('view', {})
    layer_name = view.get('view_layer') or target.get('view_layer')
    view_layer = scene.view_layers.get(layer_name) if layer_name else scene.view_layers[0]
    if view_layer is None:
        raise ValueError('Native capture view layer is absent from snapshot')
    kind = target['owner_kind'].lower()
    for layer in scene.view_layers:
        # A compositor Render Layers node selects its own view layer, which
        # need not match the node editor's selected layer. Shader AOV capture
        # still belongs to exactly the frozen viewport/camera layer.
        layer.use = kind == 'compositor' or layer == view_layer
    if kind == 'compositor':
        # Render Layers can point at a different scene from the compositor's
        # owner. Ensure its requested layer remains renderable in this helper.
        for source_scene in bpy.data.scenes:
            for layer in source_scene.view_layers:
                layer.use = True
    if bpy.context.window is not None:
        bpy.context.window.view_layer = view_layer
    mark('scene_selection')
    restored = _restore_images(inputs.get('images', ()))
    _restore_clips(inputs.get('clips', ()))
    mark('resource_restoration')
    companion = _companion(inputs, restored, context)
    if companion is not None:
        mark('companion_sampling')
        if measured:
            timings.update(worker_finished_at=time.perf_counter(), total_seconds=time.perf_counter()-started,
                           sample_bytes=companion['rgba'].nbytes)
            companion['_capture_timings'] = timings
        return companion
    from .capture_models import materialize
    prepared_provenance = materialize(inputs.get('_pg_models', ()), context)
    mark('frozen_grade_preparation')
    _disable_file_outputs()
    mark('output_suppression')
    token = hashlib.sha256(repr(inputs.get('identity')).encode()).hexdigest()[:16]
    if kind in ('material', 'world'):
        if view.get('width') and view.get('height'):
            scene.render.resolution_x = int(view['width'])
            scene.render.resolution_y = int(view['height'])
            scene.render.resolution_percentage = 100
        color, mask = _shader(scene, view_layer, target, view, token)
    elif kind == 'compositor':
        color = mask = None
    else:
        raise ValueError('Native scopes support Material, World, and Compositor Color inputs')
    mark('shader_tap_and_camera')
    _checkpoint(context)
    artifact_directory = None
    if bake_mode:
        raw_directory = inputs.get('artifact_dir')
        if not isinstance(raw_directory, str):
            raise ValueError('Compositor bake has no artifact directory')
        artifact_directory = Path(raw_directory)
        if (not artifact_directory.is_absolute() or artifact_directory.is_symlink()
                or not artifact_directory.is_dir()):
            raise ValueError('Invalid compositor bake artifact directory')
        # A retried attempt starts only after its preceding helper retired.
        # Its private output files must not be confused with this new capture.
        for previous in artifact_directory.glob('pg_bake_*.exr'):
            previous.unlink()

    # Scope output is private to this invocation and can be removed as soon as
    # it has been sampled.  Bake output belongs to the foreground lease and is
    # intentionally left in ``artifact_directory`` until release_bake().
    temporary_context = (nullcontext(artifact_directory) if bake_mode else
                         tempfile.TemporaryDirectory(prefix='perceptual-grade-native-scope-'))
    with temporary_context as temporary:
        temporary = Path(temporary)
        if kind == 'compositor':
            (_bake_compositor if bake_mode else _compositor)(scene, target, token, temporary)
        _settings(scene, kind != 'compositor')
        # No external output writes; every readback lives in this private helper.
        scene.render.filepath = str(Path(temporary) / 'unused')
        mark('compositor_tap_and_output_setup')
        result = bpy.ops.render.render(write_still=False, scene=scene.name,
                                       layer='' if kind == 'compositor' else view_layer.name)
        mark('native_render')
        if 'FINISHED' not in result:
            raise RuntimeError('Native scope evaluation did not complete')
        _checkpoint(context)
        if bake_mode:
            outputs = tuple(temporary.glob('pg_bake_*.exr'))
            if len(outputs) > 1:
                raise RuntimeError('Native compositor bake did not produce one image')
            if outputs:
                output = outputs[0]
            else:
                # Constant Color outputs have no intrinsic image domain for a
                # File Output node. The connected root output evaluates them on
                # the compositor canvas, so preserve that Render Result.
                image = bpy.data.images.get('Render Result')
                if image is None:
                    raise RuntimeError('Native compositor bake produced no pixels')
                output = temporary / 'pg_bake_constant.exr'
                image.save_render(str(output), scene=scene)
            from ...scopes.exr import create as create_exr_reader
            exr_reader = create_exr_reader(None)
            with output.open('rb') as stream:
                _, (xmin, ymin, xmax, ymax), _ = exr_reader._header(stream)
            width, height = xmax - xmin + 1, ymax - ymin + 1
            if width < 1 or height < 1:
                raise ValueError('Native compositor bake has an empty pixel domain')
            if width * height * 16 > 512 * 1024 * 1024:
                raise MemoryError('Native compositor bake exceeds the 512 MiB pixel budget')
            # Blender supplies the full native input raster. Only the selected
            # Grade/Stack is then evaluated analytically, with no LUT lookup.
            from ....perceptual_grade_core.jobs.codec import thaw_model
            from ....perceptual_grade_core.model import raster
            analytic = inputs.get('analytic', {})
            if analytic.get('version') != raster.SCHEMA or analytic.get('kind') not in ('GRADE', 'STACK'):
                raise ValueError('Unsupported analytic compositor bake model')
            model = thaw_model(analytic['model'])
            image = bpy.data.images.load(str(output), check_existing=False)
            try:
                image.colorspace_settings.name = 'Non-Color'
                image.alpha_mode = 'CHANNEL_PACKED'
                if hasattr(image, 'use_half_precision'):image.use_half_precision = False
                loaded_size = tuple(int(value) for value in image.size)
                if loaded_size != (width, height) or image.channels != 4:
                    raise ValueError('Native compositor bake is not RGBA')
                pixels = np.empty((height, width, 4), dtype=np.float32)
                image.pixels.foreach_get(pixels.reshape(-1))
                diagnostics = raster.grade_rgba(pixels, model, analytic['rows'],
                    stack=analytic['kind'] == 'STACK', checkpoint=context.checkpoint,
                    progress=lambda value: context.progress(.2 + .75 * value))
                _checkpoint(context)
                image.pixels.foreach_set(pixels.reshape(-1))
                image.update()
                graded_output = temporary / 'pg_bake_analytic.exr'
                image.save_render(str(graded_output), scene=scene)
            finally:
                bpy.data.images.remove(image)
            output.unlink()
            output = graded_output
            _checkpoint(context)
            return dict(path=str(output), sha256=_hash_file(output),
                        width=width, height=height, domain=(width, height),
                        identity=inputs.get('identity'), scene=target['scene'],
                        frame=int(scene.frame_current),
                        frame_subframe=float(scene.frame_subframe),
                        engine=str(scene.render.engine), node=target['node_name'],
                        socket=target.get('socket_name', 'Color'),
                        label='Compositor bake', backend=scene.render.engine,
                        evaluation='ANALYTIC', analytic_kind=analytic['kind'],
                        analytic_signature=analytic['signature'], diagnostics=diagnostics)

        maximum = tuple(int(value) for value in inputs.get('sample_size', (256, 192)))
        if len(maximum) != 2 or min(maximum) < 1 or max(maximum) > 4096:
            raise ValueError('Invalid native scope sample dimensions')
        layout=_sample_layout(inputs.get('sample_grid'))
        if kind == 'compositor':
            outputs = tuple(Path(temporary).glob('pg_scope_capture_*.exr'))
            if len(outputs) > 1:
                raise RuntimeError('Native compositor scope did not produce one captured input image')
            if outputs:
                output = outputs[0]
            else:
                image = bpy.data.images.get('Render Result')
                if image is None:
                    raise RuntimeError('Native compositor scope produced no captured values')
                output = Path(temporary) / 'constant.exr'
                image.save_render(str(output), scene=scene)
            mark('exr_output_selection_and_save')
            rgba, domain = _read_compositor(output, maximum, bool(inputs.get('exact_size', False)),layout)
        else:
            image = bpy.data.images.get('Render Result')
            if image is None:
                raise RuntimeError('Native shader scope evaluation produced no render result')
            output = Path(temporary) / 'capture.exr'
            image.save_render(str(output), scene=scene)
            mark('exr_output_selection_and_save')
            rgba, domain = _read_shader(output, view_layer.name, color, mask, maximum,
                                        bool(inputs.get('exact_size', False)),layout)
            if not np.any(rgba[..., 3] > 1e-8):
                raise ValueError('The selected ' + kind + ' has no visible samples in this view')
        mark('pixel_readback_and_sampling')
    _checkpoint(context)
    rgba.setflags(write=False)
    result = dict(rgba=rgba, identity=inputs.get('identity'), domain=domain,
                label='Compositor input' if kind == 'compositor' else 'Visible ' + kind + ': ' + target['owner_name'],
                backend=scene.render.engine, prepared_provenance=prepared_provenance)
    if measured:
        mark('result_cleanup')
        timings.update(worker_finished_at=time.perf_counter(),
                       total_seconds=time.perf_counter()-started,
                       images=len(inputs.get('images', ())), clips=len(inputs.get('clips', ())),
                       sample_bytes=rgba.nbytes)
        result['_capture_timings'] = timings
    return result
