"""Resolve a scope's native graph occurrence and view into value-only metadata.

No editor, node, scene or matrix RNA is retained by this component. Window and
area pointers are discovery hints only and are checked against live membership
before use. A nested datablock is not an evaluation context: its exact group
instance path must be selected when more than one occurrence is possible.
"""
from collections.abc import Mapping
from ..integration.components import Component


def _pointer(value):
    try:
        return int(value.as_pointer()) if value is not None else 0
    except (AttributeError, ReferenceError, RuntimeError):
        return 0


def _same(a, b):
    pointer = _pointer(a)
    return bool(pointer) and pointer == _pointer(b)


def _matrix(value):
    return tuple(tuple(float(item) for item in row) for row in value)


def group_paths(root, target):
    """All acyclic node-instance paths, including the empty root occurrence."""
    result = []
    pending = [(root, (), set())]
    try:
        while pending:
            tree, path, seen = pending.pop()
            if _same(tree, target):
                result.append(path)
                continue
            key = _pointer(tree)
            if not key or key in seen:
                continue
            ancestors = seen | {key}
            for node in reversed(tuple(tree.nodes)):
                child = getattr(node, 'node_tree', None)
                if child is not None:
                    pending.append((child, path + (str(node.name),), ancestors))
        return tuple(result)
    finally:
        pending.clear()


def editor_path(space, root, target):
    """Recover the selected occurrence without treating a shared group as unique."""
    if space is None or getattr(space, 'type', None) != 'NODE_EDITOR':
        return None
    path = tuple(getattr(item, 'node_tree', None) for item in getattr(space, 'path', ()))
    if not path or not _same(path[0], root) or not _same(path[-1], target):
        return None
    names = []
    for parent, child in zip(path, path[1:]):
        choices = [node for node in parent.nodes if _same(getattr(node, 'node_tree', None), child)]
        active = getattr(parent.nodes, 'active', None)
        if active in choices:
            chosen = active
        elif len(choices) == 1:
            chosen = choices[0]
        else:
            raise ValueError('Choose the node-group instance to sample in the node editor')
        names.append(str(chosen.name))
    return tuple(names)


def create(_application):
    import bpy
    from types import SimpleNamespace

    _last_views = {}

    def clear():
        _last_views.clear()

    def remember(context=None, event=None):
        """Record a live 3D region when it is used, including routed mouse events."""
        context = context or bpy.context
        window = getattr(context, 'window', None)
        if window is None:
            return
        area = getattr(context, 'area', None)
        if event is not None and hasattr(event, 'mouse_x') and hasattr(event, 'mouse_y'):
            x, y = event.mouse_x, event.mouse_y
            area = next((item for item in window.screen.areas
                         if item.type == 'VIEW_3D' and item.x <= x < item.x + item.width
                         and item.y <= y < item.y + item.height), None)
        if area is None or area.type != 'VIEW_3D':
            return
        region = getattr(context, 'region', None)
        if event is not None and hasattr(event, 'mouse_x') and hasattr(event, 'mouse_y'):
            region = next((item for item in area.regions
                           if item.type == 'WINDOW' and item.x <= x < item.x + item.width
                           and item.y <= y < item.y + item.height), region)
        if (region is None or region.type != 'WINDOW'
                or not any(_same(region, item) for item in area.regions)):
            region = next((item for item in area.regions if item.type == 'WINDOW'), None)
        if region is not None:
            _last_views[_pointer(window)] = (_pointer(area), _pointer(region))

    def _view_region(context):
        window = getattr(context, 'window', None)
        if window is None:
            return None
        remember(context)
        choices = []
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            for region in area.regions:
                if region.type == 'WINDOW' and region.width > 0 and region.height > 0:
                    choices.append((area, region, space))
        remembered = _last_views.get(_pointer(window))
        selected = next((item for item in choices
                         if (_pointer(item[0]), _pointer(item[1])) == remembered), None)
        if selected is None and choices:
            selected = max(choices, key=lambda item: item[1].width * item[1].height)
        return selected

    def _owner(node, context, frozen=None):
        tree = node.id_data
        kind = getattr(tree, 'bl_idname', '')
        candidates = []
        if kind == 'CompositorNodeTree':
            for scene in bpy.data.scenes:
                root = getattr(scene, 'compositing_node_group', None)
                if root is not None and group_paths(root, tree):
                    candidates.append(('compositor', scene, root, scene))
        elif kind == 'ShaderNodeTree':
            for owner_kind, values in (('material', bpy.data.materials), ('world', bpy.data.worlds),
                                       ('light', bpy.data.lights)):
                for owner in values:
                    root = getattr(owner, 'node_tree', None)
                    if root is not None and group_paths(root, tree):
                        candidates.append((owner_kind, owner, root, None))
        else:
            raise ValueError('Scopes require a shader or compositor node tree')
        if not candidates:
            return None  # A detached test/library tree has no scene evaluation domain.
        space = getattr(context, 'space_data', None)
        root = getattr(space, 'node_tree', None)
        editor_id = getattr(space, 'id', None)
        preferred = ([item for item in candidates if _pointer(item[2]) == frozen['root_pointer']]
                     if frozen and frozen.get('_lightweight') else [])
        if not preferred:preferred = [item for item in candidates if _same(item[1], editor_id)]
        if not preferred and root is not None:
            preferred = [item for item in candidates if _same(item[2], root)]
        scene = getattr(context, 'scene', None)
        if not preferred and kind == 'CompositorNodeTree':
            preferred = [item for item in candidates if _same(item[3], scene)]
        if not preferred and kind == 'ShaderNodeTree':
            obj = getattr(context, 'object', None)
            material = getattr(obj, 'active_material', None)
            preferred = [item for item in candidates if _same(item[1], material)]
            if not preferred and scene is not None:
                preferred = [item for item in candidates if _same(item[1], scene.world)]
        chosen = preferred or candidates
        if len(chosen) != 1:
            raise ValueError('Choose the material, world or scene containing this node in the node editor')
        owner_kind, owner, root, scene_owner = chosen[0]
        if owner_kind == 'light':
            raise ValueError('Light shader scopes have no visible surface or background sampling domain')
        scene = scene_owner or scene
        if scene is None:
            raise ValueError('Scopes need a scene evaluation context')
        if owner_kind == 'world' and not _same(scene.world, owner):
            raise ValueError('The selected World is not used by the current scene')
        path = tuple(frozen['group_path']) if frozen and frozen.get('_lightweight') else editor_path(space, root, tree)
        if frozen and frozen.get('_lightweight'):
            selected_tree=root
            for name in path:
                instance=selected_tree.nodes.get(name)
                selected_tree=getattr(instance,'node_tree',None)
                if selected_tree is None:raise ReferenceError('The scope node-group occurrence was removed')
            if not _same(selected_tree,tree):raise ReferenceError('The scope node-group occurrence changed')
        if path is None:
            paths = group_paths(root, tree)
            if len(paths) != 1:
                raise ValueError('Choose the node-group instance to sample in the node editor')
            path = paths[0]
        return owner_kind, owner, scene, path

    def source_context(node, context=None):
        """Freeze an editor occurrence without discovering a native scene domain.

        Image sampling does not require a material user, camera, or even a scene
        owner. The editor path is sufficient; native ownership is resolved only
        when an explicit capture asks for it.
        """
        if isinstance(context, Mapping):
            value=dict(context)
            if value.get('_lightweight') and value.get('scene_uid'):
                scene=next((item for item in bpy.data.scenes if int(item.session_uid)==value['scene_uid']),None)
                if scene is None:raise ReferenceError('The scope scene was removed')
                value['scene']=str(scene.name)
            if value.get('_lightweight') and value.get('group_pointers'):
                roots=tuple(bpy.data.node_groups)+tuple(owner.node_tree
                    for collection in (bpy.data.materials,bpy.data.worlds,bpy.data.lights)
                    for owner in collection if owner.node_tree is not None)
                tree=next((item for item in roots if _pointer(item)==value['root_pointer']),None)
                names=[]
                for pointer in value['group_pointers']:
                    instance=next((item for item in tree.nodes if _pointer(item)==pointer),None) if tree else None
                    if instance is None or getattr(instance,'node_tree',None) is None:
                        raise ReferenceError('The scope node-group occurrence was removed')
                    names.append(str(instance.name));tree=instance.node_tree
                value['group_path']=tuple(names)
            return value
        context = context or bpy.context
        tree = node.id_data
        space = getattr(context, 'space_data', None)
        root = getattr(space, 'node_tree', None)
        path = editor_path(space, root, tree) if root is not None else None
        if path is None:
            root, path = tree, ()
        editor = tuple(_pointer(getattr(context, name, None)) for name in ('window', 'area', 'region'))
        scene = getattr(context, 'scene', None)
        token = (int(getattr(root, 'session_uid', 0)), _pointer(root))
        tree=root;pointers=[]
        for name in path:
            instance=tree.nodes.get(name)
            if instance is None:raise ReferenceError('The scope node-group occurrence was removed')
            pointers.append(_pointer(instance));tree=instance.node_tree
        return dict(_lightweight=True, consumer=(editor + (token, tuple(pointers))) if getattr(space,'type',None)=='NODE_EDITOR' else None,
                    root_pointer=_pointer(root), tree_pointer=_pointer(node.id_data), group_path=path, group_pointers=tuple(pointers),
                    scene=str(scene.name) if scene is not None else '',scene_uid=int(scene.session_uid) if scene is not None else 0,
                    editor=editor)

    def consumer(node, context=None):
        return source_context(node, context)['consumer']

    def occurrence(node, context=None):
        """Return temporary parent instances for one already frozen occurrence."""
        value = source_context(node, context)
        path = tuple(value.get('group_path', ()))
        if not path:
            return ()
        if value.get('_lightweight'):
            pointer = value['root_pointer']
            roots = tuple(bpy.data.node_groups) + tuple(
                owner.node_tree for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights)
                for owner in collection if owner.node_tree is not None)
            root = next((tree for tree in roots if _pointer(tree) == pointer), None)
        elif value.get('owner_kind') == 'compositor':
            root = bpy.data.scenes[value['scene']].compositing_node_group
        else:
            collection = bpy.data.materials if value.get('owner_kind') == 'material' else bpy.data.worlds
            root = collection[value['owner_name']].node_tree
        if root is None:
            raise ReferenceError('The scope node-tree occurrence was removed')
        parents = []
        for name in path:
            instance = root.nodes.get(name)
            if instance is None or getattr(instance, 'node_tree', None) is None:
                raise ReferenceError('The scope node-group occurrence was removed')
            parents.append(instance)
            root = instance.node_tree
        if not _same(root, node.id_data):
            raise ReferenceError('The scope node-group occurrence changed')
        return tuple(parents)

    def thaw_context(value):
        """Resolve the frozen editor; never substitute another active editor."""
        if not isinstance(value, Mapping) or not value.get('_lightweight'):
            return value or bpy.context
        wp, ap, rp = value['editor']
        if not any((wp, ap, rp)):
            return SimpleNamespace(scene=bpy.data.scenes.get(value.get('scene', '')),
                                   window=None, area=None, region=None, space_data=None,
                                   object=None, view_layer=None)
        window = next((item for item in bpy.context.window_manager.windows if _pointer(item) == wp), None)
        area = next((item for item in window.screen.areas if _pointer(item) == ap), None) if window else None
        region = next((item for item in area.regions if _pointer(item) == rp), None) if area else None
        if window is None or area is None or region is None:
            raise ReferenceError('The scope editor was closed')
        return SimpleNamespace(window=window, area=area, region=region, space_data=area.spaces.active,
                               scene=window.scene, view_layer=window.view_layer,
                               object=window.view_layer.objects.active)

    def _view(scene, context, compositor=False):
        current_layer = getattr(context, 'view_layer', None)
        layer = scene.view_layers.get(getattr(current_layer, 'name', '')) or scene.view_layers[0]
        base = dict(view_layer=str(layer.name), frame=int(scene.frame_current),
                    frame_subframe=float(scene.frame_subframe), engine=str(scene.render.engine))
        if compositor:
            percentage = scene.render.resolution_percentage / 100.
            base.update(width=max(1, int(scene.render.resolution_x * percentage)),
                        height=max(1, int(scene.render.resolution_y * percentage)), source='compositor')
            return base
        selected = _view_region(context)
        if selected is not None:
            area, region, space = selected
            rv3d = space.region_3d
            quadviews = tuple(getattr(space, 'region_quadviews', ()))
            regions = tuple(item for item in area.regions if item.type == 'WINDOW')
            if quadviews and region in regions:
                index = regions.index(region)
                if index < len(quadviews):
                    rv3d = quadviews[index]
            if rv3d is not None:
                visible = tuple(sorted(str(obj.name) for obj in scene.objects
                                       if obj.visible_get(view_layer=layer, viewport=space)))
                base.update(source='viewport', width=int(region.width), height=int(region.height),
                            view_matrix=_matrix(rv3d.view_matrix),
                            projection_matrix=_matrix(rv3d.window_matrix),
                            clip_start=float(space.clip_start), clip_end=float(space.clip_end),
                            perspective=str(rv3d.view_perspective), visible_objects=visible,
                            viewport=(_pointer(area), _pointer(region)))
                return base
        camera = scene.camera
        if camera is None:
            raise ValueError('Shader scopes need a 3D viewport or an active scene camera; '
                             'open a 3D view in this window or assign a camera to scene '
                             + repr(scene.name))
        percentage = scene.render.resolution_percentage / 100.
        base.update(source='camera', camera=str(camera.name),
                    width=max(1, int(scene.render.resolution_x * percentage)),
                    height=max(1, int(scene.render.resolution_y * percentage)))
        return base

    def describe(node, socket_name='Color', context=None):
        """Freeze ownership and the current view; no native value escapes."""
        frozen = context if isinstance(context, Mapping) else None
        if isinstance(context, Mapping) and not context.get('_lightweight'):
            result = dict(context)
            result['node_name'] = str(node.name)
            result['socket_name'] = str(socket_name)
            return result
        context = thaw_context(context)
        selected = _owner(node, context, frozen)
        if selected is None:
            return None
        kind, owner, scene, path = selected
        if node.inputs.get(socket_name) is None:
            raise ValueError('The scope input socket no longer exists: ' + socket_name)
        return dict(consumer=frozen['consumer'] if frozen else consumer(node, context), scene=str(scene.name), owner_kind=kind,
                    owner_name=str(owner.name), group_path=path, node_name=str(node.name),
                    socket_name=str(socket_name), view=_view(scene, context, kind == 'compositor'))

    return Component({'__name__': __name__, 'describe': describe, 'consumer': consumer,
                      'remember': remember, 'clear': clear, 'group_paths': group_paths,
                      'editor_path': editor_path, 'source_context': source_context,
                      'occurrence': occurrence, 'thaw_context': thaw_context})
