"""Materialize frozen grading dependencies exclusively inside a capture helper.

The caller has loaded an immutable scene snapshot. No App, live output receipt,
or foreground resource is consulted or modified by this operation.
"""
import numpy as np

from ....perceptual_grade_core.jobs.codec import thaw_model
from ....perceptual_grade_core.model import lut


ROWS = ('Scene to LUT R', 'Scene to LUT G', 'Scene to LUT B',
        'LUT to Scene R', 'LUT to Scene G', 'LUT to Scene B')
LAYOUT = 2
KERNEL_SCHEMA = 25


def _cube(record, encoded, context):
    from ...model import stack_baking
    model = thaw_model(record['model'])
    if lut.signature(model) != record['cube_signature']:
        raise ValueError('Frozen capture model signature does not match its receipt')
    baker = stack_baking.CPU(model, progressive=False, encoded=encoded)
    checkpoint = baker.checkpoint

    def check(progress=None):
        context.checkpoint()
        checkpoint(progress)

    baker.checkpoint = check
    while baker.cube is None:
        context.checkpoint()
        baker.advance()
    cube = baker.cube
    if cube.size != 65 or cube.signature != record['cube_signature']:
        raise ValueError('Capture dependency did not produce its exact final cube')
    return cube


def _root(record):
    import bpy
    kind = record['owner_kind']
    if kind == 'compositor':
        scene = bpy.data.scenes.get(record['scene'])
        root = getattr(scene, 'compositing_node_group', None)
    else:
        collection = {'material': bpy.data.materials, 'world': bpy.data.worlds,
                      'light': bpy.data.lights}.get(kind)
        owner = collection.get(record['owner_name']) if collection is not None else None
        root = getattr(owner, 'node_tree', None)
    if root is None:
        raise ValueError('Frozen grading dependency owner is absent from the snapshot')
    return root


def _target(record, copies):
    root = _root(record)
    identity = (record['owner_kind'], record.get('owner_name', ''), record['scene'])
    path = ()
    for name in record['group_path']:
        path += (name,)
        key = identity + (path,)
        group = root.nodes.get(name)
        if group is None or getattr(group, 'node_tree', None) is None:
            raise ValueError('Frozen grading dependency group occurrence is absent')
        if key not in copies:
            group.node_tree = group.node_tree.copy()
            copies.add(key)
        root = group.node_tree
    node = root.nodes.get(record['node_name'])
    if node is None or node.node_tree is None:
        raise ValueError('Frozen grading dependency node is absent')
    return node


def materialize(records, context):
    """Return the final provenance actually installed in the isolated graph."""
    import bpy
    copies, visited, cubes, provenance = set(), set(), {}, set()
    for record in records:
        context.checkpoint()
        identity = (record['owner_kind'], record.get('owner_name', ''), record['scene'],
                    tuple(record['group_path']), record['node_name'])
        if identity in visited:
            raise ValueError('Frozen grading dependency occurrence is duplicated')
        visited.add(identity)
        node = _target(record, copies)
        original = node.node_tree
        if original.get('_pg_output_layout') != LAYOUT or original.get('_pg_cube_schema') != KERNEL_SCHEMA:
            raise ValueError('Unsupported capture dependency kernel layout')
        rows = np.asarray(record['rows'], dtype=np.float32)
        if rows.shape != (2, 3, 3) or not np.isfinite(rows).all():
            raise ValueError('Invalid frozen capture endpoint matrices')
        encoded = original.bl_idname == 'CompositorNodeTree'
        cache_key = (record['cube_signature'], encoded)
        cube = cubes.get(cache_key)
        if cube is None:
            cube = _cube(record, encoded, context)
            cubes[cache_key] = cube
        else:
            # A reused numerical artifact still requires this record to bind
            # the same authored model, rather than trusting its claimed key.
            if lut.signature(thaw_model(record['model'])) != cube.signature:
                raise ValueError('Frozen capture model signature does not match its receipt')
        pixels = cube.packed_pixels() if encoded else cube.pixels()
        if not np.isfinite(pixels).all():
            raise ValueError('Nonfinite frozen capture payload')
        context.checkpoint()
        kernel = original.copy()
        image = None
        try:
            fields = {str(item['_pg_matrix_row']): item for item in kernel.nodes
                      if item.get('_pg_matrix_row')}
            selectors = [item for item in kernel.nodes if item.get('_pg_output_selector')]
            endpoints = [item for item in kernel.nodes if item.get('_pg_cube_image')
                         and item.get('_pg_output_slot') == 2]
            if set(fields) != set(ROWS) or len(selectors) != 1 or not endpoints:
                raise ValueError('Incomplete private capture kernel')
            image = bpy.data.images.new('Frozen capture LUT', width=lut.ATLAS_WIDTH,
                height=pixels.shape[0], alpha=True, float_buffer=True)
            image.colorspace_settings.name = 'Non-Color'
            image.alpha_mode = 'CHANNEL_PACKED'
            image.use_half_precision = False
            image.pixels.foreach_set(np.ascontiguousarray(pixels).ravel())
            image.update()
            image['_pg_cube_schema'] = KERNEL_SCHEMA
            image['_pg_cube_size'] = 65
            image['_pg_cube_encoding'] = 'RGBE_DIFFERENCES' if encoded else 'RGB'
            image['_pg_cube_signature'] = cube.signature
            image['_pg_output_slot'] = 2
            for endpoint in endpoints:
                endpoint.image = image
            for name, values in zip(ROWS, rows.reshape(6, 3)):
                for socket, value in zip(fields[name].inputs, values):
                    socket.default_value = float(value)
            selectors[0].inputs[0].default_value = 2.
            selectors[0].inputs[1].default_value = 0.
            context.checkpoint()
            node.node_tree = kernel
            provenance.add((cube.signature, cube.size))
        except BaseException:
            if node.node_tree == kernel:
                node.node_tree = original
            bpy.data.node_groups.remove(kernel)
            if image is not None and image.users == 0:
                bpy.data.images.remove(image)
            raise
    return tuple(sorted(provenance))
