"""Blender-native scene adapters around fixed, unclipped grading RGB."""

from ..integration.components import Component

def create(_application):
    from functools import lru_cache
    import numpy as np
    from mathutils import Color
    from ...perceptual_grade_core.model import linear

    LUT_INPUT_ROWS = ('Scene to LUT R', 'Scene to LUT G', 'Scene to LUT B')
    LUT_OUTPUT_ROWS = ('LUT to Scene R', 'LUT to Scene G', 'LUT to Scene B')


    @lru_cache(maxsize=3)
    def adapters(space):
        """Probe the active OCIO conversion, including its Bradford adaptation.

        Only called on Blender's main thread. The inverse of the same sampled
        matrix avoids compounding independently rounded native conversions.
        """
        working_space = _application.component('graph.nodes').working_space
        if space != working_space():
            raise ValueError('Scene adapter must be sampled in the active working space')
        if space == 'Linear Rec.709':
            return np.eye(3), np.eye(3)
        basis = np.eye(3)
        forward = np.array([tuple(Color(row).from_scene_linear_to_rec709_linear())
                            for row in basis]).T
        if not np.isfinite(forward).all() or np.linalg.cond(forward) > 1e12:
            raise ValueError('The active OCIO scene conversion is not an invertible RGB matrix')
        return forward, np.linalg.inv(forward)


    ROW = '_pg_matrix_row'


    def constants(tree, names):
        for name, row in zip(names, linear.I3):
            node = tree.nodes.new('ShaderNodeCombineXYZ')
            node[ROW] = name
            node.label = name
            for socket, value in zip(node.inputs, row):
                socket.default_value = value


    def row_node(tree, name):
        matches = [node for node in tree.nodes if node.get(ROW) == name]
        if (len(matches) != 1 or matches[0].bl_idname != 'ShaderNodeCombineXYZ'
                or any(socket.is_linked for socket in matches[0].inputs)):
            raise ValueError('Invalid private matrix row')
        return matches[0]


    def read_rows(node, names=None):
        names = (*LUT_INPUT_ROWS, *LUT_OUTPUT_ROWS) if names is None else names
        return tuple(tuple(socket.default_value for socket in row_node(node.node_tree, name).inputs)
                     for name in names)


    def write_rows(node, rows):
        nodes = _application.component('graph.nodes')
        for name, row in zip((*LUT_INPUT_ROWS, *LUT_OUTPUT_ROWS), rows):
            for socket, value in zip(row_node(node.node_tree, name).inputs, row):
                nodes.set_uniform(socket, value)


    def multiply(tree, rgb, names):
        dots = []
        for name in names:
            node = tree.nodes.new('ShaderNodeVectorMath'); node.operation = 'DOT_PRODUCT'
            node.label = name
            tree.links.new(rgb, node.inputs[0])
            tree.links.new(row_node(tree, name).outputs[0], node.inputs[1])
            dots.append(node.outputs['Value'])
        combined = tree.nodes.new('ShaderNodeCombineXYZ')
        for value, socket in zip(dots, combined.inputs):
            tree.links.new(value, socket)
        return combined.outputs['Vector']


    def uniforms(node, before, after):
        lut = _application.component('graph.lut')
        return lut.commit_endpoint_rows(node, before, after)


    def _set_export_LUT_INPUT_ROWS(value):
        nonlocal LUT_INPUT_ROWS
        LUT_INPUT_ROWS = value

    def _set_export_LUT_OUTPUT_ROWS(value):
        nonlocal LUT_OUTPUT_ROWS
        LUT_OUTPUT_ROWS = value

    def _set_export_adapters(value):
        nonlocal adapters
        adapters = value

    def _set_export_write_rows(value):
        nonlocal write_rows
        write_rows = value

    def _set_export_multiply(value):
        nonlocal multiply
        multiply = value

    def _set_export_uniforms(value):
        nonlocal uniforms
        uniforms = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Color': lambda: Color,
        'LUT_INPUT_ROWS': lambda: LUT_INPUT_ROWS,
        'LUT_OUTPUT_ROWS': lambda: LUT_OUTPUT_ROWS,
        'adapters': lambda: adapters,
        'constants': lambda: constants,
        'row_node': lambda: row_node,
        'read_rows': lambda: read_rows,
        'write_rows': lambda: write_rows,
        'linear': lambda: linear,
        'lru_cache': lambda: lru_cache,
        'multiply': lambda: multiply,
        'np': lambda: np,
        'uniforms': lambda: uniforms,
    }, writers={
        'LUT_INPUT_ROWS': _set_export_LUT_INPUT_ROWS,
        'LUT_OUTPUT_ROWS': _set_export_LUT_OUTPUT_ROWS,
    'adapters': _set_export_adapters, 'write_rows': _set_export_write_rows, 'multiply': _set_export_multiply, 'uniforms': _set_export_uniforms})
