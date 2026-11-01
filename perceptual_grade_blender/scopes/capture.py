"""App-owned native graph captures and immutable snapshot leases.

Only this foreground adapter reads live RNA. The dedicated capture helper owns
its loaded scene; numerical workers continue to receive arrays and values.
"""
from ..integration.components import Component


def create(application):
    import hashlib
    import logging
    from pathlib import Path
    import shutil
    import re
    import time
    import uuid
    from contextlib import nullcontext
    from collections import OrderedDict, deque
    from collections.abc import Mapping
    from itertools import count
    from ..runtime.protocol import Protocol
    import bpy
    import numpy as np
    from ...blender_runtime.execution.contracts import JobSpec

    states = OrderedDict()
    batches = OrderedDict()
    retiring_batches = []
    retiring = []
    bakes = {}
    bake_retiring = []
    revisions = {}
    image_probes = OrderedDict()
    geometry_probes = OrderedDict()
    DEBOUNCE = .3
    MAX_BYTES = 512 * 1024 * 1024
    MAX_CAPTURES = 8
    MAX_BAKES = 8
    generation = 0
    input_generation = 1
    update_epoch = 1
    processed_epochs = {}
    capture_requests = count(1)
    input_requests = count(1)
    diagnostic_events = deque(maxlen=65536)
    diagnostics_enabled = False
    diagnostic_dropped = 0
    ownership=application.component('ownership')
    PG_NODE_TYPES = {'PGCompositorGrade', 'PGCompositorStack', 'PGShaderGrade', 'PGShaderStack'}

    def enable_diagnostics(enabled=True):
        """Opt in to bounded value-only timings; never change capture admission."""
        nonlocal diagnostics_enabled, diagnostic_dropped
        diagnostics_enabled = bool(enabled)
        diagnostic_events.clear()
        diagnostic_dropped = 0
        for state in (*states.values(), *bakes.values(), *retiring, *bake_retiring):
            state.pop('_diagnostic_wait', None)

    def _diagnostic_value(value, depth=0):
        if value is None or type(value) in (str, bool, int, float):return value
        if depth >= 12:return '<diagnostic-depth-limit>'
        try:
            if isinstance(value, Mapping):
                return {str(key):_diagnostic_value(item,depth+1) for key,item in value.items()}
            if isinstance(value, (tuple,list)):
                return tuple(_diagnostic_value(item,depth+1) for item in value)
        except Exception:
            pass
        return '<'+type(value).__name__+'>'

    def diagnostics():
        records = [_diagnostic_value(record) for record in diagnostic_events]
        if diagnostic_dropped:
            records.append(dict(event='overflow', at=time.perf_counter(), count=diagnostic_dropped))
        return records

    def _trace(event, request=None, **values):
        nonlocal diagnostic_dropped
        if not diagnostics_enabled:return
        try:
            record = dict(event=event, at=time.perf_counter(), **values)
            if request is not None:
                record.update(capture=_fingerprint(request.get('identity', ())),
                              node=request.get('node_name', ''), owner_kind=request.get('owner_kind', ''),
                              socket=request.get('socket_name', ''), scene=request.get('scene', ''))
            record = _diagnostic_value(record)
            if len(diagnostic_events) == diagnostic_events.maxlen:diagnostic_dropped += 1
            diagnostic_events.append(record)
        except Exception:
            # Instrumentation must never affect capture admission or leases.
            diagnostic_dropped += 1

    def _stamp():
        return time.perf_counter() if diagnostics_enabled else None

    def _phase(started, name, request, **values):
        if started is not None:
            try:_trace('phase', request, phase=name, seconds=time.perf_counter()-started, **values)
            except Exception:pass

    def _wait(state, reason=None):
        if not diagnostics_enabled:return
        try:
            previous = state.get('_diagnostic_wait')
            if previous is not None and previous[0] == reason:return
            if previous is not None:
                _trace('wait', state['request'], reason=previous[0],
                       seconds=time.perf_counter()-previous[1])
            if reason is None:state.pop('_diagnostic_wait', None)
            else:state['_diagnostic_wait'] = (reason, time.perf_counter())
        except Exception:
            # Waiting and retirement remain correct if diagnostics fail.
            state.pop('_diagnostic_wait', None)

    def _trace_submit(runtime, request, attempt, operation, lane='capture'):
        if not diagnostics_enabled:return
        try:
            helper = runtime.helpers.get(lane)
            values = dict(helper_ready=bool(helper is not None and helper.ready),
                          helper_generation=getattr(helper,'generation',0))
        except Exception as error:
            values = dict(helper_metadata_error=type(error).__name__)
        _trace('submit', request, attempt=attempt, operation=operation, **values)

    def _hash(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _value(value):
        if value is None or type(value) in (str, bool, int, float):
            return value
        if isinstance(value,bpy.types.ID):
            resource=None
            if getattr(value,'filepath',''):
                try:
                    original,paths=_resource_paths(value.filepath,getattr(value,'source','FILE'))
                    resource=(str(original),tuple((str(path),path.stat().st_size,path.stat().st_mtime_ns)
                                                 for path in paths))
                except FileNotFoundError:resource=('missing',str(value.filepath))
            return (value.bl_rna.identifier,int(value.session_uid),
                    value.library.filepath if value.library else '',resource)
        try:
            return tuple(_value(v) for v in value)
        except (TypeError, ReferenceError):
            return type(value).__name__

    def _properties(value, seen=None, depth=0, ignored_properties=()):
        """Freeze settings generically, including ramps, curves and node storage."""
        ignored = {'rna_type', 'name', 'name_full', 'node_tree', 'id_data', 'original',
                   'evaluated_get'}
        ignored.update(ignored_properties)
        if isinstance(value,bpy.types.Node):
            ignored.update({'location','location_absolute','width','height','dimensions','select','show_options',
                            'show_preview','show_texture','hide','label','color','use_custom_color',
                            'parent','is_registered_node_type','internal_links','inputs','outputs',
                            'pg_bake_image'})
            if value.bl_idname not in PG_NODE_TYPES:
                # These extension settings are registered on every Blender Node,
                # but native Math/Image/etc. operations never evaluate them.
                ignored.update({'pg_linear', 'pg_tone', 'pg_chromatic'})
        if value.bl_rna.identifier in {'MeshVertex','MeshEdge','MeshPolygon','MeshUVLoop'}:
            ignored.update({'select','select_edge','hide','pin_uv'})
        if isinstance(value,bpy.types.Object):
            ignored.update({'hide_select','show_name','show_axis','show_texture_space','show_wire',
                'show_all_edges','show_bounds','display_type','display_bounds_type','active_material_index',
                'select_get','users_collection','users_scene'})
        if isinstance(value,bpy.types.ViewLayer):
            ignored.update({'depsgraph','objects','active_layer_collection'})
        if isinstance(value,bpy.types.RenderSettings):
            ignored.update({'filepath','image_settings','file_extension','use_file_extension',
                            'use_placeholder','use_overwrite','stamp_note_text'})
        seen=set() if seen is None else seen
        try:pointer=value.as_pointer()
        except AttributeError:return _value(value)
        chromatic_settings = value.bl_rna.identifier == 'PGChromaticSettings'
        if pointer in seen or depth>8:return ('reference',type(value).__name__)
        seen=seen|{pointer}
        result = []
        for prop in value.bl_rna.properties:
            if prop.identifier in ignored:continue
            # Unsaved sidebar aliases compute display values by scanning their
            # owner. The corresponding saved fields below contain the meaning.
            if chromatic_settings and prop.identifier.startswith('sidebar_'):continue
            if prop.is_readonly and prop.type not in {'POINTER','COLLECTION'}:continue
            try:
                item=getattr(value,prop.identifier)
                if prop.type=='COLLECTION':
                    frozen=tuple(_properties(entry,seen,depth+1) for entry in item)
                elif prop.type=='POINTER' and item is not None and not isinstance(item,bpy.types.ID):
                    frozen=_properties(item,seen,depth+1)
                else:frozen=_value(item)
                result.append((prop.identifier,frozen))
            except (AttributeError,TypeError,ValueError,ReferenceError,RuntimeError):
                continue
        return tuple(result)

    def _node_watch(node, output_name, incoming):
        if node.bl_idname in PG_NODE_TYPES and output_name=='Baked Color':
            kernel=getattr(node,'node_tree',None)
            image_node=next((item for item in kernel.nodes if item.get('_pg_baked_image')),None) if kernel else None
            return ('baked',_value(getattr(image_node,'image',None)))
        live_pg=node.bl_idname in PG_NODE_TYPES and output_name in {'','Color'}
        semantic=application.component('graph.lut').scope_signature(node) if live_pg else _properties(node)
        return (semantic,
                tuple((socket.identifier,_value(getattr(socket,'default_value',None)),
                       tuple((link.from_node.name,link.from_socket.identifier,bool(link.is_muted))
                             for link in incoming(socket))) for socket in node.inputs
                      if not live_pg or socket.name in ({'Color','Alpha'} if not output_name else {'Color'})),
                tuple((socket.identifier,_value(getattr(socket,'default_value',None))) for socket in node.outputs
                      if not live_pg or socket.name in {'Color','Alpha'}))

    def _graph(node=None, socket_name='Color', whole_tree=None, skip=None, watch=None,
               parents=(), manifest=None, owner=None, bake_watch=False):
        """Reach only the requested socket through exact group occurrences.

        Unknown native nodes conservatively consume all their inputs. Group
        instances consume only sockets reached through their active output.
        """
        records=[]; structure=[]; images=set(); seen=set(); incoming={}
        lut=application.component('graph.lut')
        def links(socket):
            tree=socket.id_data;key=tree.as_pointer()
            if key not in incoming:
                table={}
                for link in tree.links:
                    if not link.is_muted and link.is_valid:
                        table.setdefault(link.to_socket.as_pointer(),[]).append(link)
                incoming[key]=table
            return incoming[key].get(socket.as_pointer(),())
        def matching(sockets,socket):
            return next((item for item in sockets if item.identifier==socket.identifier),sockets.get(socket.name))
        if whole_tree is None:
            work=[('input',node.inputs[socket_name],tuple(parents))]
        else:
            work=[('input',socket,()) for endpoint in whole_tree.nodes
                  if endpoint.bl_idname in {'ShaderNodeOutputMaterial','ShaderNodeOutputWorld',
                    'ShaderNodeOutputLight','NodeGroupOutput','CompositorNodeComposite'}
                  and getattr(endpoint,'is_active_output',True) for socket in endpoint.inputs]
        try:
            while work:
                direction,socket,path=work.pop()
                if socket is None:continue
                source=socket.node
                key=(source.id_data.session_uid,source.as_pointer(),socket.identifier,
                     tuple(item.as_pointer() for item in path),direction)
                if key in seen:continue
                seen.add(key)
                if direction=='input':
                    edges=links(socket)
                    structure.append((key,tuple((link.from_node.as_pointer(),link.from_socket.identifier) for link in edges)))
                    if edges:work.extend(('output',link.from_socket,path) for link in reversed(edges))
                    else:records.append((key,_value(getattr(socket,'default_value',None))))
                    continue
                structure.append((key,source.bl_idname,bool(source.mute)))
                if skip is not None and (source.id_data.as_pointer(),source.as_pointer())==skip[:2] and (
                        len(skip)==2 or (tuple(item.name for item in path)==skip[2] and (len(skip)<4 or owner is not None and (owner['owner_kind'],owner['owner_name'])==skip[3]))):
                    records.append((key,'selected-terminal'));continue
                if source.mute:
                    bypass=tuple(link.from_socket for link in source.internal_links if link.to_socket==socket)
                    if bypass:
                        work.extend(('input',item,path) for item in reversed(bypass));continue
                kind=source.bl_idname
                if kind=='NodeReroute':
                    work.append(('input',source.inputs[0],path));continue
                if kind=='NodeGroupInput' and path:
                    work.append(('input',matching(path[-1].inputs,socket),path[:-1]));continue
                is_pg=kind in PG_NODE_TYPES and not lut.is_adopted(source)
                if bake_watch:
                    is_pg = (application.component('graph.nodes').is_grade(source)
                             or application.component('graph.stacks').is_stack(source))
                if is_pg and socket.name in {'Baked Color','Baked Alpha'}:
                    image=application.component('graph.bake').baked_image(source)
                    if image is not None:images.add(image.name_full)
                    if not bake_watch:
                        records.append((key,'baked',_value(image)))
                        continue
                    kernel=getattr(source,'node_tree',None)
                    sampler=next((item for item in kernel.nodes if item.get('_pg_baked_image')),
                                 None) if kernel is not None else None
                    sampler_vector=None if sampler is None else sampler.inputs.get('Vector')
                    sampler_state=(None if sampler is None else
                        (tuple((name,getattr(sampler,name,None)) for name in
                               ('interpolation','extension','projection','projection_blend')),
                         tuple((link.from_node.as_pointer(),link.from_socket.identifier)
                               for link in sampler_vector.links) if sampler_vector else (),
                         _value(sampler_vector.default_value) if sampler_vector else None))
                    records.append((key,'baked',_value(image),sampler_state))
                    vector=source.inputs.get('Vector')
                    if vector is not None:
                        work.append(('input',vector,path))
                    continue
                if is_pg and socket.name=='Alpha':
                    work.append(('input',source.inputs.get('Alpha'),path));continue
                if is_pg and socket.name=='Color':
                    if bake_watch:
                        nodes = application.component('graph.nodes')
                        intent = nodes.output_intent(source)
                        records.append((key, intent, nodes.observe_intent(source, intent)))
                        work.append(('input', source.inputs.get('Color'), path))
                        continue
                    model=application.component('graph.nodes').bake_model(source)
                    rows=lut.scope_rows(source,model)
                    records.append((key,lut.scope_signature(source,model,rows)))
                    if manifest is not None and owner is not None:
                        from ...perceptual_grade_core.jobs.codec import freeze_model
                        manifest.setdefault('models',[]).append(dict(owner,
                            group_path=tuple(item.name for item in path),node_name=source.name,
                            model=freeze_model(model),rows=tuple(tuple(tuple(float(v) for v in row) for row in matrix) for matrix in rows),
                            cube_signature=lut.cm.signature(model)))
                    work.append(('input',source.inputs.get('Color'),path));continue
                child=getattr(source,'node_tree',None)
                if child is not None:
                    outputs=tuple(item for item in child.nodes if item.bl_idname=='NodeGroupOutput')
                    active=next((item for item in outputs if getattr(item,'is_active_output',False)),outputs[0] if outputs else None)
                    if active is None:raise ValueError('Scope source group has no active output')
                    if any(item.node_tree==child for item in path):raise ValueError('Scope source has a recursive node group')
                    work.append(('input',matching(active.inputs,socket),path+(source,)))
                    records.append((key,kind,bool(source.mute)));continue
                image=getattr(source,'image',None)
                if image is not None:images.add(image.name_full)
                if kind=='CompositorNodeRLayers' and manifest is not None:
                    scene=getattr(source,'scene',None)
                    if scene is None and owner is not None:scene=bpy.data.scenes.get(owner['scene'])
                    if scene is not None:manifest.setdefault('scenes',{})[scene.as_pointer()]=scene
                records.append((key,kind,_properties(source),
                    tuple((output.identifier,_value(getattr(output,'default_value',None))) for output in source.outputs)))
                if watch is not None:
                    watch[(int(source.id_data.session_uid),source.name,source.as_pointer(),socket.name)]=_node_watch(source,socket.name,links)
                work.extend(('input',item,path) for item in reversed(tuple(source.inputs)))
            if manifest is not None:manifest.setdefault('structure',[]).extend(structure)
            return tuple(records),images
        finally:
            work.clear();incoming.clear();seen.clear()

    def _ancestor_inputs(context, scene, watch=None):
        """A nested tree's Group Input gets values from its exact outer instances."""
        kind=context['owner_kind']
        owner=(scene if kind=='compositor' else
               (bpy.data.materials if kind=='material' else bpy.data.worlds).get(context['owner_name']))
        tree=(scene.compositing_node_group if kind=='compositor' else owner.node_tree)
        values=[];images=set()
        for name in context.get('group_path',()):
            instance=tree.nodes.get(name)
            if instance is None or getattr(instance,'node_tree',None) is None:
                raise ReferenceError('The scope node-group instance was removed')
            inputs=[]
            for socket in instance.inputs:
                graph,names=_graph(instance,socket.name,watch=watch)
                inputs.append((socket.identifier,graph));images.update(names)
            values.append((name,_properties(instance),tuple(inputs)))
            tree=instance.node_tree
        return tuple(values),images

    def _referenced_scenes(tree):
        """Find Render Layers scene IDs through nested compositor groups."""
        result = []; seen = set(); work = [('tree', tree)]
        try:
            while work:
                kind, value = work.pop()
                if kind == 'tree':
                    if value is None or value.as_pointer() in seen:continue
                    seen.add(value.as_pointer())
                    work.extend(('node', node) for node in reversed(tuple(value.nodes)))
                else:
                    scene = getattr(value, 'scene', None)
                    if scene is not None:result.append(scene)
                    child = getattr(value, 'node_tree', None)
                    if child is not None:work.append(('tree', child))
            return tuple(dict((scene.as_pointer(), scene) for scene in result).values())
        finally:
            work.clear()
            result.clear()

    def _scene_revision(scene,watch=None):
        shaders,images=_scene_shaders(scene,watch=watch)
        return (scene.name_full,scene.frame_current,scene.frame_subframe,
                scene.render.engine,_properties(scene.render),
                _properties(scene.cycles) if hasattr(scene,'cycles') else (),
                _properties(scene.eevee) if hasattr(scene,'eevee') else (),
                shaders),images

    def _mesh_signature(mesh):
        """Hash semantic mesh arrays in bulk without per-element RNA traversal."""
        ignored = {'vertices', 'edges', 'loops', 'polygons', 'loop_triangles',
                   'loop_triangle_polygons', 'uv_layer_clone', 'uv_layer_clone_index',
                   'uv_layer_stencil', 'uv_layer_stencil_index',
                   'corner_normals', 'vertex_normals', 'polygon_normals',
                   'uv_layers', 'attributes', 'color_attributes', 'vertex_colors'}
        digest = hashlib.sha256()
        digest.update(_fingerprint(_properties(mesh, ignored_properties=ignored)).encode())

        def array(values, name, size, kind):
            data = np.empty(len(values) * size, dtype=kind)
            values.foreach_get(name, data)
            digest.update(memoryview(data).cast('B'))

        array(mesh.polygons, 'loop_start', 1, np.int32)
        array(mesh.polygons, 'loop_total', 1, np.int32)
        # Position, edge/corner topology, material indices, normals, UVs and
        # shader/geometry-node attributes all live in Mesh.attributes.
        for attribute in mesh.attributes:
            name = attribute.name
            if name.startswith(('.select_', '.hide_', '.uv_select_', '.uv_pin')):
                continue
            digest.update(repr((name, attribute.domain, attribute.data_type,
                                len(attribute.data))).encode())
            if not attribute.data:
                continue
            for prop in attribute.data[0].bl_rna.properties:
                if prop.identifier == 'rna_type':
                    continue
                if prop.type in {'FLOAT', 'INT', 'BOOLEAN'}:
                    size = prop.array_length if prop.is_array else 1
                    kind = np.float32 if prop.type == 'FLOAT' else np.int32 if prop.type == 'INT' else np.bool_
                    digest.update(prop.identifier.encode())
                    array(attribute.data, prop.identifier, size, kind)
                elif prop.type == 'STRING':
                    digest.update(_fingerprint(tuple(getattr(value, prop.identifier)
                                                     for value in attribute.data)).encode())
        digest.update(repr(tuple((layer.name, bool(layer.active_render))
                                 for layer in mesh.uv_layers)).encode())
        return digest.hexdigest()

    def _geometry_signature(obj, *, fresh=False):
        data=obj.data
        if data is None:return None
        if fresh:
            # Bake observations can run immediately before publication, when
            # no foreground depsgraph poll has advanced the scope revisions.
            # Their request owns an exact read of relevant geometry and UVs.
            return _mesh_signature(data) if isinstance(data, bpy.types.Mesh) else _fingerprint(_properties(data))
        key=(int(data.session_uid),revisions.get(('data',data.as_pointer()),0),
             revisions.get(('object',obj.as_pointer()),0))
        value=geometry_probes.get(key)
        if value is None:
            value=_fingerprint(_properties(data))
            geometry_probes[key]=value
            while len(geometry_probes)>128:geometry_probes.popitem(last=False)
        return value

    def _scene_shaders(scene,skip=None,watch=None,manifest=None,view=None,bake_watch=False):
        """The source render scene, including off-camera lighting dependencies."""
        objects=tuple(obj for obj in scene.objects if not obj.hide_render or obj==scene.camera)
        if view and view.get('source')=='viewport':
            visible=set(view.get('visible_objects',()))
            objects=tuple(obj for obj in scene.objects if obj.name in visible or obj==scene.camera)
        owners=[]
        if scene.world is not None:owners.append(('world',scene.world))
        for obj in objects:
            if isinstance(obj.data,bpy.types.Light):owners.append(('light',obj.data))
            owners.extend(('material',slot.material) for slot in obj.material_slots if slot.material is not None)
        values=[];seen=set();images=set()
        for kind,owner in owners:
            key=owner.as_pointer()
            if key in seen:continue
            seen.add(key)
            tree=getattr(owner,'node_tree',None)
            root=dict(owner_kind=kind,owner_name=owner.name_full,scene=scene.name_full)
            graph,names=(_graph(whole_tree=tree,skip=skip,watch=watch,manifest=manifest,owner=root,
                                bake_watch=bake_watch)
                         if tree is not None else (None,()))
            values.append((owner.bl_rna.identifier,int(owner.session_uid),_properties(owner),graph))
            images.update(names)
        object_values=tuple((int(obj.session_uid),_properties(obj),
                       tuple(tuple(row) for row in obj.matrix_world),
                       _geometry_signature(obj,fresh=bake_watch)) for obj in objects)
        render=(scene.frame_current,scene.frame_subframe,_properties(scene.render),
                _properties(scene.cycles) if hasattr(scene,'cycles') else (),
                _properties(scene.eevee) if hasattr(scene,'eevee') else (),
                tuple((layer.name,_properties(layer)) for layer in scene.view_layers))
        return (tuple(sorted(values,key=lambda item:item[:2])),object_values,render),images

    def _resource_paths(filepath,source):
        path=Path(bpy.path.abspath(filepath))
        pattern=path.name
        if source in {'SEQUENCE','TILED'}:
            pattern=pattern.replace('<UDIM>','*').replace('<UVTILE>','*')
            if pattern==path.name:pattern=re.sub(r'\d+(?=\D*$)','*',pattern)
            paths=tuple(sorted(item for item in path.parent.glob(pattern) if item.is_file()))
        else:paths=(path,) if path.is_file() else ()
        if not paths:raise FileNotFoundError('Native scope resource is unavailable: '+str(path))
        return path,paths

    def _copy_resource(value,directory,runtime,number):
        original,paths=_resource_paths(value.filepath,getattr(value,'source','FILE'))
        target=directory/('resource-'+str(number));target.mkdir(mode=0o700)
        files=[]
        for source in paths:
            before=source.stat()
            if before.st_size+_lease_bytes(runtime)>MAX_BYTES:
                raise MemoryError('Native scope resource snapshot budget exhausted')
            copied=target/source.name
            shutil.copy2(source,copied)
            after=source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('A native scope resource changed while it was being copied')
            files.append(dict(path=str(copied),sha256=_hash(copied)))
        primary=next((item for item in files if Path(item['path']).name==original.name),files[0])
        return dict(primary,name=value.name_full,source='file',files=tuple(files),
                    filepath=str(target/original.name))

    def _require_snapshot_image(image):
        if ((image.is_dirty or image.source=='GENERATED') and
                (image.source=='TILED' or image.use_multiview or image.type=='MULTILAYER')):
            raise ValueError('Native scope capture cannot snapshot edited tiled, multiview or multilayer Image buffers')

    def _image_content(image,native_lease=False):
        """Probe relevant volatile pixels because Image.update emits no ID update."""
        _require_snapshot_image(image)
        if image.get('_pg_cube_signature'):
            return image.get('_pg_cube_signature'),image.get('_pg_cube_size')
        if not (image.is_dirty or image.source=='GENERATED'):return None
        key=image.as_pointer();now=time.monotonic();old=image_probes.get(key)
        if old is not None and old[0]==update_epoch and now-old[1]<.5:return old[2]
        width,height=image.size;count=width*height*image.channels
        if count*4>MAX_BYTES:raise MemoryError('Native scope image snapshot budget exhausted')
        if count<1 or not image.has_data:return ('empty',tuple(image.size))
        guard=application.component('graph.output_guard')
        if not native_lease and not guard.available():
            if old is not None:return old[2]
            raise application.component('graph.lut').ScopePreparing('Waiting to read the scope image')
        with nullcontext() if native_lease else guard.native_operation():
            pixels=np.empty(count,dtype=np.float32);image.pixels.foreach_get(pixels)
        digest=hashlib.sha256(memoryview(pixels).cast('B')).hexdigest()
        image_probes[key]=(update_epoch,now,digest)
        image_probes.move_to_end(key)
        while len(image_probes)>128:image_probes.popitem(last=False)
        return digest

    def _image_state(names,native_lease=False):
        result = []
        for name in sorted(names):
            image = bpy.data.images.get(name)
            if image is None:
                continue
            _require_snapshot_image(image)
            path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
            stat = None
            if path is not None:
                try:
                    _,files=_resource_paths(image.filepath,image.source)
                    stat=tuple((str(item),item.stat().st_size,item.stat().st_mtime_ns) for item in files)
                except FileNotFoundError:stat=('missing',str(path))
            result.append((int(image.session_uid), tuple(image.size), image.colorspace_settings.name,
                                image.alpha_mode, image.source, bool(image.is_dirty), stat,
                                _image_content(image,native_lease)))
        return tuple(sorted(result,key=lambda item:item[0]))

    def _contains(value, item):
        work = [value]; seen = set()
        try:
            while work:
                current = work.pop()
                if isinstance(current, str):
                    if current == item:return True
                elif isinstance(current, (tuple, list)) and id(current) not in seen:
                    seen.add(id(current));work.extend(reversed(current))
            return False
        finally:
            work.clear()

    def _fingerprint(value):
        """Hash Python's container representation without recursive repr calls.

        Frozen graph records use scalar values, tuples, lists and dictionaries.
        Preserve their existing bytes, including singleton tuples, dict order
        and Python's recursive-container sentinels.
        """
        digest = hashlib.sha256(); work = [('value', value)]; active = set()
        try:
            while work:
                kind, current = work.pop()
                if kind == 'text':
                    digest.update(current.encode());continue
                if kind == 'close':
                    pointer, ending = current
                    active.remove(pointer);digest.update(ending.encode());continue
                if isinstance(current, (tuple, list, dict)):
                    pointer = id(current)
                    if pointer in active:
                        digest.update(b'(...)' if isinstance(current, tuple) else
                                      b'[...]' if isinstance(current, list) else b'{...}')
                        continue
                    active.add(pointer)
                    if isinstance(current, dict):
                        digest.update(b'{');work.append(('close', (pointer, '}')))
                        items = tuple(current.items())
                        for index in range(len(items) - 1, -1, -1):
                            key, child = items[index]
                            work.extend((('value', child), ('text', ': '), ('value', key)))
                            if index:work.append(('text', ', '))
                    else:
                        sequence = isinstance(current, tuple)
                        digest.update(b'(' if sequence else b'[')
                        ending = ',)' if sequence and len(current) == 1 else ')' if sequence else ']'
                        work.append(('close', (pointer, ending)))
                        for index in range(len(current) - 1, -1, -1):
                            work.append(('value', current[index]))
                            if index:work.append(('text', ', '))
                else:
                    digest.update(repr(current).encode())
            return digest.hexdigest()
        finally:
            work.clear()
            active.clear()

    def _settled_shaders(node,context,value):
        """Acquisition freezes the current shading inputs immediately.

        Retained accepted captures provide feedback while a new capture runs;
        replacing this fingerprint with an older settled digest would authorize
        a different graph than the one the user just requested.
        """
        return _fingerprint(value),False

    def _poll_updates(scene,view):
        """Observe each scene's depsgraph only after a callback signals work."""
        current=update_epoch;key=(scene.as_pointer(),view.get('view_layer'))
        if processed_epochs.get(key)==current:return
        guard=application.component('graph.output_guard')
        if not guard.available():return
        try:
            layer=scene.view_layers.get(view.get('view_layer','')) or scene.view_layers[0]
            with bpy.context.temp_override(scene=scene,view_layer=layer):
                invalidate(depsgraph=bpy.context.evaluated_depsgraph_get())
            processed_epochs[key]=current
        except (AttributeError,ReferenceError,RuntimeError):
            pass

    def _image_metadata(names):
        result=[]
        for name in sorted(names):
            image=bpy.data.images.get(name)
            if image is None:result.append((name,None));continue
            _require_snapshot_image(image)
            path=Path(bpy.path.abspath(image.filepath)) if image.filepath else None
            try:stat=(path.stat().st_size,path.stat().st_mtime_ns) if path else None
            except OSError:stat='missing'
            result.append((name,int(image.session_uid),tuple(image.size),str(image.filepath),
                           image.source,image.colorspace_settings.name,image.alpha_mode,
                           bool(image.is_dirty),stat,revisions.get(('image',image.as_pointer()),0),
                           _image_content(image)))
        return tuple(result)

    def _scene_watch(scene, surfaces):
        settings=(int(scene.session_uid),scene.frame_current,scene.frame_subframe,
                  scene.render.engine,_properties(scene.render))
        if not surfaces:return settings
        return settings+(int(scene.world.session_uid) if scene.world else 0,
                         tuple((int(obj.session_uid),obj.name_full,bool(obj.hide_render),
                                tuple(tuple(row) for row in obj.matrix_world),
                                int(obj.data.session_uid) if obj.data else 0,
                                tuple(int(slot.material.session_uid) if slot.material else 0 for slot in obj.material_slots),
                                revisions.get(('object',obj.as_pointer()),0),
                                revisions.get(('data',obj.data.as_pointer()),0) if obj.data else 0)
                               for obj in scene.objects))

    def _view_watch(request):
        view=request['view'];scene=bpy.data.scenes.get(request['scene'])
        if scene is None:return None
        if view.get('source')=='camera':return ('camera',int(scene.camera.session_uid) if scene.camera else 0)
        if view.get('source')!='viewport':return ('compositor',)
        if 'viewport' not in view:
            return ('frozen',view.get('view_matrix'),view.get('projection_matrix'),view['width'],view['height'])
        ap,rp=view['viewport']
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.as_pointer()!=ap or area.type!='VIEW_3D':continue
                region=next((item for item in area.regions if item.as_pointer()==rp),None)
                if region is None:return None
                space=area.spaces.active;rv3d=space.region_3d
                quadviews=tuple(getattr(space,'region_quadviews',()))
                regions=tuple(item for item in area.regions if item.type=='WINDOW')
                if quadviews and region in regions and regions.index(region)<len(quadviews):
                    rv3d=quadviews[regions.index(region)]
                if rv3d is None:return None
                layer=scene.view_layers.get(view['view_layer'])
                return (region.width,region.height,tuple(tuple(row) for row in rv3d.view_matrix),
                        tuple(tuple(row) for row in rv3d.window_matrix),space.clip_start,space.clip_end,
                        str(rv3d.view_perspective),tuple(sorted(obj.name for obj in scene.objects
                            if layer is not None and obj.visible_get(view_layer=layer,viewport=space))))
        return None

    def describe(node, socket_name='Color', context=None,*,_native_lease=False):
        total_started=_stamp()
        phase_started=_stamp()
        contexts=application.component('scopes.context')
        source_context=contexts.source_context(node,context.get('_source_context',context) if isinstance(context,Mapping) else context)
        context = contexts.describe(node, socket_name, context=source_context)
        if context is None:
            return None
        _phase(phase_started,'context',context)
        phase_started=_stamp()
        watch={};manifest={}
        scene = bpy.data.scenes.get(context['scene'])
        if scene is None:
            raise ReferenceError('The scope scene was removed')
        parents=application.component('scopes.context').occurrence(node,context)
        owner={key:context[key] for key in ('owner_kind','owner_name','scene')}
        graph,names=_graph(node,socket_name,watch=watch,parents=parents,manifest=manifest,owner=owner)
        _poll_updates(scene,context['view'])
        references=tuple(manifest.get('scenes',{}).values())
        needs_scene=context['owner_kind']!='compositor' or bool(references)
        feedback=any(_contains(graph,kind) for kind in ('ShaderNodeShaderToRGB','ShaderNodeAmbientOcclusion'))
        skip=None if feedback else (node.id_data.as_pointer(),node.as_pointer(),tuple(context.get('group_path',())),(context['owner_kind'],context['owner_name']))
        needs_main=context['owner_kind']!='compositor' or scene in references
        shaders,shader_images=_scene_shaders(scene,skip,watch,manifest,context['view']) if needs_main else ((),set())
        referenced_values=[]
        for referenced in references:
            if referenced==scene:continue
            _poll_updates(referenced,context['view'])
            value,images=_scene_shaders(referenced,watch=watch,manifest=manifest)
            referenced_values.append((int(referenced.session_uid),value));shader_images.update(images)
        _phase(phase_started,'graph_and_dependency_description',context,
               upstream_images=len(names),referenced_scenes=len(referenced_values))
        phase_started=_stamp()
        shader_revision,scene_pending=_settled_shaders(node,context,
            (shaders,tuple(referenced_values),_image_state(shader_images,_native_lease))) if needs_scene else ((),False)
        # Shader resources belong to the settled shading revision; this Grade's
        # own LUT upload must not invalidate the immediate analytical response.
        image_state = _image_state(names,_native_lease)
        scene_values = ((scene.frame_current,scene.frame_subframe,
                         scene.render.engine,_properties(scene.render),
                         _properties(scene.cycles) if hasattr(scene,'cycles') else (),
                         _properties(scene.eevee) if hasattr(scene,'eevee') else (),
                         shader_revision) if needs_scene else
                        (scene.frame_current,scene.frame_subframe,scene.render.engine,
                         _properties(scene.render)))
        _phase(phase_started,'resource_and_scene_revision',context,images=len(names),
               scene_images=len(shader_images),scene_pending=bool(scene_pending))
        phase_started=_stamp()
        source_view={key:value for key,value in context['view'].items() if key not in {'viewport','camera','visible_objects','view_layer'}}
        source_view['view_layer']=scene.view_layers[context['view']['view_layer']].as_pointer()
        if 'camera' in context['view']:source_view['camera']=int(scene.camera.session_uid) if scene.camera else 0
        if 'visible_objects' in context['view']:
            source_view['visible_objects']=tuple(sorted(int(scene.objects[name].session_uid) for name in context['view']['visible_objects'] if name in scene.objects))
        version = _fingerprint((graph, image_state, scene_values, source_view, generation))
        result = dict(context,_target_pointer=node.as_pointer())
        result['_source_context']=source_context
        result['image_names'] = tuple(sorted(names))
        # Occurrence and graph wiring are hard boundaries. Frame, parameter and
        # camera/view motion within them may publish completed progress.
        result['continuity']=_fingerprint((context['consumer'],int(scene.session_uid),
            context['owner_kind'],
            tuple(item.as_pointer() for item in parents),node.as_pointer(),socket_name,
            tuple(manifest.get('structure',())),context['view'].get('source'),
            context['view'].get('viewport'),source_view['view_layer'],generation))
        unique={}
        for record in manifest.get('models',()):
            key=(record['owner_kind'],record['owner_name'],record['scene'],record['group_path'],record['node_name'])
            unique[key]=record
        result['_pg_models']=tuple(unique.values())
        result['_scope_provenance']=tuple(sorted({(record['cube_signature'],65) for record in unique.values()}))
        result['scene_pending'] = scene_pending
        result['identity'] = ('native-scope-v1', context['consumer'], version, socket_name)
        result['version'] = version
        result['_watch_nodes']=tuple(watch.items())
        result['_watch_scene']=_scene_watch(scene,needs_scene)
        result['_watch_images']=_image_metadata(names|shader_images)
        result['_watch_needs_scene']=needs_scene
        result['_watch_view']=_view_watch(context)
        if diagnostics_enabled:
            result['_watch_parts']=dict(graph=_fingerprint(graph),images=_fingerprint(image_state),
                                        scene=_fingerprint(scene_values),view=_fingerprint(source_view))
        _phase(phase_started,'identity_hash',result)
        _phase(total_started,'describe_total',result)
        return result

    def describe_source(node,socket_name='Color',context=None,route=None,*,_native_lease=False):
        route=route or application.component('scopes.sources').resolve(node,socket_name,context)
        if route.kind=='native':return describe(node,socket_name,context)
        source_context=application.component('scopes.context').source_context(node,context)
        base=application.component('scopes.context').describe(node,socket_name,context=source_context)
        if base is None:raise ValueError('Scope source has no native acquisition context')
        result=dict(base,_source_context=source_context,_target_pointer=node.as_pointer(),_direct_kind=route.kind,
                    _scope_provenance=(),_pg_models=())
        if route.kind=='image':
            from dataclasses import asdict
            result['_image_source']=asdict(route.source)
            result['_image_key']=route.source.key
            source=(route.source.key,_image_state({route.source.name},_native_lease))
            continuity=route.source.continuity
        else:
            result['_constant_source']=route.source
            source=route.source;continuity=('constant',)
        version=_fingerprint((route.kind,source,generation))
        result.update(version=version,identity=('scope-source-v2',base['consumer'],version,socket_name),
            continuity=_fingerprint((base['consumer'],route.kind,continuity,socket_name,generation)))
        return result

    def channel(node):
        consumer = application.component('scopes.context').consumer(node)
        return node.as_pointer() if consumer is None else (node.as_pointer(), consumer)

    def _lease_bytes(runtime):
        return sum(p.stat().st_size for directory in tuple(runtime.session.path.glob('capture-*'))
                   + tuple(runtime.session.path.glob('bake-*'))
                   for p in directory.rglob('*') if p.is_file())

    def _capture_protocol(state):
        """The native snapshot owns a separate actor from analytical consumers."""
        protocol=state.get('protocol')
        if protocol is None:
            source=application.component('graph.scheduler').identity(
                ('native-snapshot',state['request']['identity']))
            protocol=Protocol(application,'scope',identity=dict(
                request=next(capture_requests),source=source,generation=1))
            protocol.require('request','capture')
            state['protocol']=protocol
        return protocol

    def _finish_capture_protocol(state):
        """Called with no worker, or after this exact helper physically retires."""
        protocol=state.get('protocol')
        if protocol is None:return
        phase=protocol.state.phase
        if phase=='capturing' or state.get('error'):protocol.event('failed')
        elif phase=='running':protocol.event('complete')
        if protocol.state.phase=='retiring':protocol.event('retired',validated=True)
        protocol.event('close')
        if protocol.state.phase=='retiring':protocol.event('retired',validated=True)
        protocol.close()
        state.pop('protocol',None)

    def _finish_input_protocol(state):
        """Release request authority only after no physical reader remains."""
        protocol=state.get('input_protocol')
        if protocol is None:return
        if protocol.state.phase=='capturing':protocol.event('failed')
        if protocol.state.phase=='retiring':protocol.event('retired',validated=True)
        protocol.event('close')
        protocol.close()
        state.pop('input_protocol',None)

    def _snapshot(state):
        request = state['request']
        total_started = _stamp()
        if diagnostics_enabled:
            state['_diagnostic_attempt'] = state.get('_diagnostic_attempt', 0) + 1
        attempt = state.get('_diagnostic_attempt', 0)
        runtime = application.component('services').get_runtime()
        if _lease_bytes(runtime) >= MAX_BYTES:
            raise MemoryError('Native scope snapshot budget exhausted')
        scene_uid=request.get('_source_context',{}).get('scene_uid')
        scene=(next((item for item in bpy.data.scenes if int(item.session_uid)==scene_uid),None)
               if scene_uid else bpy.data.scenes.get(request['scene']))
        if scene is None:
            raise ReferenceError('The scope scene was removed')
        guard = application.component('graph.output_guard')
        guard.require_available()
        prefix = 'bake-' if state.get('bake') else 'capture-'
        directory = runtime.session.path / (prefix + uuid.uuid4().hex)
        directory.mkdir(mode=0o700)
        state['directory'] = directory
        images = [];clips=[]
        _wait(state)
        _trace('snapshot_begin', request, attempt=attempt)
        outcome = 'failed'
        try:
            with guard.native_operation():
                # New scenes can have a stale depsgraph while custom output
                # groups are being serialized; update every view layer first.
                phase_started = _stamp()
                for layer in scene.view_layers:layer.update()
                _phase(phase_started, 'view_layer_update', request, attempt=attempt,
                       layers=len(scene.view_layers))
                node=ownership.resolve(state.get('node'))
                if node is None:raise ReferenceError('The native scope node was removed')
                # Admission always probes volatile pixels afresh. A user can
                # edit Image.pixels without producing any depsgraph update.
                image_probes.clear()
                phase_started = _stamp()
                current=(_describe_bake(node,request,_native_lease=True) if state.get('bake') else
                         (describe_source(node,request['socket_name'],context=request.get('_source_context'),_native_lease=True) if request.get('_direct_kind') else describe(node,request['socket_name'],context=request,_native_lease=True)))
                _phase(phase_started, 'snapshot_revalidation', request, attempt=attempt)
                if state.get('bake') and _bake_inputs_changed_state(state, native_lease=True, force=True):
                    raise BakeInputsChanged('Bake inputs changed before freezing completed')
                if current['identity']!=request['identity'] or current.get('scene_pending'):
                    _trace('snapshot_rejected', request, attempt=attempt,
                           reason='scene_pending' if current.get('scene_pending') else 'revision_changed',
                           replacement=_fingerprint(current['identity']) if diagnostics_enabled else '',
                           changed_parts=tuple(key for key,value in request.get('_watch_parts',{}).items()
                               if current.get('_watch_parts',{}).get(key)!=value),
                           changed_nodes=tuple((key[1],key[3]) for key,value in request.get('_watch_nodes',())
                               if dict(current.get('_watch_nodes',())).get(key)!=value))
                    if not state.get('bake'):
                        raise ValueError('Scope inputs changed before the snapshot was frozen')
                    raise BakeInputsChanged('Bake inputs changed before the snapshot was frozen')
                for key in ('_sample_grid','_scope_epoch'):
                    if key in request:current[key]=request[key]
                request=current;state['request']=current
                snapshot = directory / 'scene.blend'
                phase_started = _stamp()
                roots = {scene}
                if state.get('worker_operation') == 'shader_bake':
                    owner_kind = request.get('owner_kind')
                    owner_name = request.get('owner_name')
                    owner_collection = bpy.data.materials if owner_kind == 'material' else bpy.data.worlds
                    owner = owner_collection.get(str(owner_name))
                    if owner is not None:
                        roots.add(owner)
                bpy.data.libraries.write(str(snapshot), roots, path_remap='ABSOLUTE')
                _phase(phase_started, 'blend_serialization', request, attempt=attempt,
                       bytes=snapshot.stat().st_size if diagnostics_enabled else 0)
                # Packed pixels are in the blend. Generated/edited live buffers
                # are runtime data and require a separately owned immutable copy.
                dependencies={}
                phase_started = _stamp()
                for value,users in bpy.data.user_map().items():
                    for user in users:dependencies.setdefault(user,set()).add(value)
                pending=list(roots);used=set()
                while pending:
                    value=pending.pop()
                    if value in used:continue
                    used.add(value);pending.extend(dependencies.get(value,()))
                _phase(phase_started, 'snapshot_dependency_walk', request, attempt=attempt,
                       datablocks=len(used))
                phase_started = _stamp()
                buffer_bytes = 0; packed_images = 0; copied_bytes = 0; copied_files = 0
                for image in tuple(value for value in used if isinstance(value,bpy.types.Image)):
                    _require_snapshot_image(image)
                    # Only volatile buffers need a loaded pixel array. Disk files
                    # must be snapshotted even before Blender has opened them.
                    if image.is_dirty or image.source=='GENERATED':
                        width,height=image.size
                        if width<1 or height<1 or not image.has_data:
                            raise ValueError('Native scope image has no loaded pixels: '+image.name_full)
                        count=width*height*image.channels
                        if count*4+_lease_bytes(runtime)>MAX_BYTES:
                            raise MemoryError('Native scope image snapshot budget exhausted')
                        pixels=np.empty(count,dtype=np.float32);image.pixels.foreach_get(pixels)
                        buffer_bytes += count * 4
                        path=directory/('image-'+str(len(images))+'.npy')
                        np.save(path,pixels.reshape(height,width,image.channels),allow_pickle=False)
                        images.append(dict(name=image.name_full,path=str(path),sha256=_hash(path),
                                           colorspace=image.colorspace_settings.name,
                                           alpha_mode=image.alpha_mode,is_float=bool(image.is_float),source='buffer'))
                    elif image.packed_file:
                        packed_images += 1
                        continue
                    elif image.filepath:
                        record=_copy_resource(image,directory,runtime,len(images))
                        record.update(colorspace=image.colorspace_settings.name,
                                      alpha_mode=image.alpha_mode,is_float=bool(image.is_float))
                        images.append(record)
                        if diagnostics_enabled:
                            copied_files += len(record['files'])
                            copied_bytes += sum(Path(item['path']).stat().st_size for item in record['files'])
                    else:
                        raise ValueError('Native scope image resource is unavailable: '+image.name_full)
                for clip in tuple(value for value in used if isinstance(value,bpy.types.MovieClip)):
                    clips.append(_copy_resource(clip,directory,runtime,len(images)+len(clips)))
                    if diagnostics_enabled:
                        copied_files += len(clips[-1]['files'])
                        copied_bytes += sum(Path(item['path']).stat().st_size for item in clips[-1]['files'])
                _phase(phase_started, 'resource_snapshot', request, attempt=attempt,
                       buffers=sum(item.get('source')=='buffer' for item in images),
                       buffer_bytes=buffer_bytes, packed_images=packed_images,
                       copied_files=copied_files, copied_bytes=copied_bytes, clips=len(clips))
                if _lease_bytes(runtime) > MAX_BYTES:
                    raise MemoryError('Native scope snapshot budget exhausted')
            payload = {key:value for key,value in request.items()
                       if key not in ('consumer','version','_sample_grid','_scope_provenance','_source_context','_target_pointer')
                       and not key.startswith('_watch')}
            phase_started = _stamp()
            payload.update(snapshot=dict(path=str(snapshot),sha256=_hash(snapshot)),
                           images=tuple(images), clips=tuple(clips))
            _phase(phase_started, 'snapshot_hash', request, attempt=attempt)
            if diagnostics_enabled:payload['_capture_diagnostics'] = True
            if state.get('bake'):
                payload.update(_bake=True, artifact_dir=str(directory))
                operation = state.get('worker_operation', 'compositor_bake')
            else:
                payload.update(sample_size=state['size'],
                               exact_size=state.get('exact_size',False))
                if '_sample_grid' in request:payload['sample_grid']=request['_sample_grid']
                operation = 'scope_capture'
                _capture_protocol(state)
                batch=batches[state['batch']]
                if batch['protocol'].state.phase=='authorized':batch['protocol'].require('start','capture')
                state['input_protocol'].require('start','capture')
            if state.get('defer_submit'):
                # Keep the click's immutable snapshot while admission waits for
                # a worker or an earlier publication to this destination.
                state['payload'] = payload
                outcome = 'frozen'
            else:
                state['handle'] = runtime.submit(JobSpec(operation, payload,
                    lane='bake_0' if state.get('bake') else 'capture', key=None))
                outcome = 'submitted'
        except Exception:
            shutil.rmtree(directory,ignore_errors=True)
            state['directory'] = None
            if not state.get('bake'):_finish_capture_protocol(state)
            raise
        finally:
            _phase(total_started, 'snapshot_total', request, attempt=attempt, outcome=outcome)
        # Optional metadata reads happen after submission's cleanup boundary.
        if outcome == 'submitted':
            _trace_submit(runtime, request, attempt, operation,
                          'bake_0' if state.get('bake') else 'capture')

    def _dispose_files(state):
        directory = state.get('directory')
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
            state['directory'] = None

    def _dispose(state):
        _wait(state)
        _trace('capture_retired', state['request'], attempt=state.get('_diagnostic_attempt',0),
               had_snapshot=state.get('directory') is not None)
        _finish_capture_protocol(state)
        _finish_input_protocol(state)
        ownership.release(state.pop('node',None))
        _dispose_files(state)

    def _dispose_bake_files(state):
        """Release the snapshot and EXR after the foreground consumes it."""
        _dispose_files(state)
        state['directory'] = None
        for reference, _revision in state.pop('grade_inputs', ()):
            ownership.release(reference)
        ownership.release(state.pop('input_target', None))
        callback = state.get('on_retired')
        if callback is not None:
            callback()
            state.pop('on_retired', None)

    def _retire_bake(state):
        ownership.release(state.pop('node',None))
        handle = state.get('handle')
        if state not in bake_retiring:
            bake_retiring.append(state)
        if handle is not None and not handle.retired():
            handle.cancel()
        else:
            _dispose_bake_files(state)
            state['handle'] = None
            if state in bake_retiring:
                bake_retiring.remove(state)

    def _retire(state):
        _wait(state)
        _trace('capture_cancelled' if state.get('handle') is not None else 'capture_released',
               state['request'], attempt=state.get('_diagnostic_attempt',0))
        ownership.release(state.pop('node',None))
        protocol=state.get('protocol')
        if protocol is not None:protocol.event('cancel')
        protocol=state.get('input_protocol')
        if protocol is not None:protocol.event('cancel')
        handle = state.get('handle')
        if handle is not None and not handle.retired():
            handle.cancel()
            if not any(item is state for item in retiring):retiring.append(state)
        else:
            _dispose(state)

    def collect_retired():
        """Drain physically retired file leases without admitting native work."""
        for state in tuple(retiring):
            if state['handle'] is None or state['handle'].retired():
                _dispose(state); retiring.remove(state)
        for state in tuple(bake_retiring):
            # A shutdown can finish with ``closed`` set while a helper could
            # not be reaped. Only this attempt's retirement releases its files.
            if state['handle'] is None or state['handle'].retired():
                _dispose_bake_files(state)
                state['handle'] = None
                state['terminal'] = True
                state['error'] = state.get('error') or 'Compositor bake cancelled'
                bake_retiring.remove(state)
        for batch in tuple(retiring_batches):
            if _finish_batch_protocol(batch):retiring_batches.remove(batch)
        return not (retiring or bake_retiring or retiring_batches)

    def poll():
        runtime = application.component('services').current_runtime()
        if runtime is not None and not runtime.closed:
            runtime.poll(budget_ms=4)
        collect_retired()
        pending = False
        automatic=application.component('scopes.automatic')
        automatic.sync()
        for state in tuple(states.values()):
            if automatic.playing():continue
            if state.get('phase')!='capturing':continue
            handle = state.get('handle')
            if handle is not None:
                if not handle.retired():
                    _wait(state, 'helper_processing')
                    pending = True; continue
                try:
                    result = handle.result()
                    if result['identity'] != state['request']['identity']:
                        raise ValueError('Native capture returned a foreign source revision')
                    if tuple(result.get('prepared_provenance',())) != tuple(state['request'].get('_scope_provenance',())):
                        raise ValueError('Native capture prepared foreign grading dependencies')
                    rgba = np.array(result['rgba'], dtype=np.float32, copy=True)
                    if rgba.ndim != 3 or rgba.shape[2] != 4:
                        raise ValueError('Native capture returned invalid RGBA pixels')
                    rgba.flags.writeable = False
                    protocol=_capture_protocol(state)
                    protocol.require('captured','compute',payload=id(rgba),validated=True)
                    protocol.require('visual','publish_visual',payload=id(rgba),validated=True)
                    state['input_protocol'].require('captured','publish',payload=id(rgba),validated=True)
                    state['candidate'] = dict(result, rgba=rgba)
                    state['phase']='candidate'
                    _wait(state)
                    _trace('capture_received', state['request'], attempt=state.get('_diagnostic_attempt',0),
                           domain=tuple(result.get('domain', ())), samples=tuple(rgba.shape[:2]),
                           helper_timings=result.get('_capture_timings', {}))
                except Exception as error:
                    state['error'] = str(error)
                    _wait(state)
                    _trace('capture_error', state['request'], attempt=state.get('_diagnostic_attempt',0),
                           error_type=type(error).__name__, error=str(error)[:240])
                state['handle'] = None
                _dispose(state)
                if state.get('error'):_fail_batch(state['batch'],state['error'])
                else:_adopt_batch(state['batch'])
            if state.get('phase')!='capturing' or state.get('error'):
                continue
            if time.monotonic() < state['due']:
                _wait(state, 'debounce')
                pending = True; continue
            if application.app.fenced or application.app.closed:
                _wait(state, 'application_fenced')
                continue
            if not application.component('graph.output_guard').available():
                _wait(state, 'native_output_guard')
                pending = True; continue
            _wait(state)
            try:
                batch=batches.get(state['batch'])
                prepare=batch.get('prepare')
                if prepare is not None:
                    try:prepare()
                    except application.component('graph.lut').ScopePreparing:
                        _wait(state,'upstream_final_output');pending=True;continue
                    batch.pop('prepare',None)
                if state.get('primary_grid'):
                    primary=next((item for item in batch['members'] if item['request']['socket_name']=='Color'),None)
                    if primary is not None:
                        candidate=primary.get('candidate')
                        if candidate is None:pending=True;continue
                        state['size']=(candidate['rgba'].shape[1],candidate['rgba'].shape[0])
                        grid=candidate.get('layout') or candidate.get('sample_grid')
                        if grid is None:
                            from ...perceptual_grade_core.scopes.samples import ordinary_layout
                            layout=ordinary_layout(*state['size'])
                            grid={name:getattr(layout,name) for name in ('mass','x','uv','tiles','ids','key')}
                        state['request']['_sample_grid']={name:grid[name] for name in ('mass','x','uv','tiles','ids','key')}
                    state['primary_grid']=False
                if state.get('payload') is not None:
                    payload=state.pop('payload')
                    payload.update(sample_size=state['size'])
                    if '_sample_grid' in state['request']:payload['sample_grid']=state['request']['_sample_grid']
                    state['handle']=runtime.submit(JobSpec('scope_capture',payload,lane='capture',key=None))
                else:_snapshot(state)
                pending = True
            except Exception as error:
                state['error'] = str(error)
                _wait(state)
                _trace('capture_error', state['request'], attempt=state.get('_diagnostic_attempt',0),
                       error_type=type(error).__name__, error=str(error)[:240])
                _fail_batch(state['batch'],state['error'])
        # Bake jobs use dedicated lanes and a different result/lifetime
        # contract: the EXR remains leased after helper retirement until the
        # foreground packs it into a Blender Image and calls release_bake().
        for token, state in tuple(bakes.items()):
            if state.get('error') or state.get('terminal'):
                continue
            handle = state.get('handle')
            if handle is not None:
                if not handle.retired():
                    pending = True
                    continue
                try:
                    result = handle.result()
                    if result.get('identity') != state['request']['identity']:
                        raise ValueError('Native bake returned a foreign source revision')
                    path = Path(result.get('path', ''))
                    directory = Path(state['directory'])
                    if (not path.is_absolute() or path.parent != directory
                            or not path.is_file() or path.is_symlink()):
                        raise ValueError('Native compositor bake returned an invalid artifact')
                    if _hash(path) != result.get('sha256'):
                        raise ValueError('Native compositor bake artifact changed after capture')
                    state['result'] = dict(result, path=str(path))
                except Exception as error:
                    state['error'] = str(error)
                state['handle'] = None
                state['terminal'] = True
                ownership.release(state.pop('node',None))
                if state.get('error'):
                    _dispose_bake_files(state)
                continue
            if state.get('result') is not None or state.get('error'):
                continue
            if state.get('defer_submit') and state.get('payload') is not None:
                # A queued click is frozen. Never read the newer live scene.
                pending = True
                continue
            if time.monotonic() < state['due']:
                pending = True
                continue
            if application.app.fenced or application.app.closed:
                continue
            if not application.component('graph.output_guard').available():
                pending = True
                continue
            try:
                node = ownership.resolve(state.get('node'))
                if node is None:
                    raise ReferenceError('The compositor bake node was removed')
                _snapshot(state)
                pending = True
            except Exception as error:
                state['error'] = str(error)
                state['terminal'] = True
                ownership.release(state.pop('node',None))
                _dispose_bake_files(state)
        bake_work = any(state.get('handle') is not None or not state.get('terminal')
                        for state in bakes.values())
        return .05 if pending or retiring or bake_retiring or bake_work else None

    def _target(node, context=None):
        editing=application.component('graph.editing')
        if editing.is_target(node):node=node.stack
        if isinstance(context,Mapping):context=context.get('_source_context',context)
        value=application.component('scopes.context').source_context(node,context)
        consumer=value.get('consumer')
        if consumer is None:
            scene=bpy.data.scenes.get(value.get('scene',''))
            consumer=('headless',int(node.id_data.session_uid),int(scene.session_uid) if scene else 0,tuple(value.get('group_pointers',value.get('group_path',()))))
        return tuple(consumer),node.as_pointer()

    def _input_state(node, socket_name='Color', context=None):
        target=_target(node,context)
        return next((value for owner,value in reversed(tuple(states.items()))
                     if owner[:2]==target and owner[2]==socket_name),None)

    def _fail_batch(target, message, stale=False):
        batch=batches.get(target)
        if batch is None:return
        batch.update(phase='stale' if stale else 'error',error=str(message))
        complete=batch.pop('complete',None)
        if complete is not None:complete(False)
        protocol=batch.get('protocol')
        if protocol is not None:protocol.event('invalidate' if stale else 'failed')
        ownership.release(batch.pop('node',None))
        for state in batch['members']:
            state['error']='' if stale else str(message)
            state['stale']=True;state['phase']='stale' if stale else 'error'
            state.pop('candidate',None)
            protocol=state.get('input_protocol')
            if protocol is not None:protocol.event('invalidate' if stale else 'failed')
            _retire(state)
        if not _finish_batch_protocol(batch) and not any(item is batch for item in retiring_batches):
            retiring_batches.append(batch)

    def _finish_batch_protocol(batch):
        protocol=batch.get('protocol')
        if any(item.get('handle') is not None and not item['handle'].retired() for item in batch['members']):
            return False
        if protocol is not None:
            if protocol.state.physical:protocol.require('retired','release',validated=True)
            protocol.event('close');protocol.close();batch.pop('protocol',None)
        batch.pop('prepare',None)
        release=batch.pop('release',None)
        if release is not None:release()
        return True

    def _adopt_batch(target):
        batch=batches.get(target)
        if batch is None or batch['phase']!='capturing':return
        if not all(item.get('candidate') is not None for item in batch['members']):return
        try:
            node=ownership.resolve(batch.get('node'))
            if node is None:raise ReferenceError('The native scope target was removed')
            automatic=application.component('scopes.automatic')
            if not all(automatic.current(item['request']) for item in batch['members']):
                raise ValueError('Scope acquisition belongs to a retired playback epoch')
            # Observe the latest semantic demand before deciding compatibility.
            complete=batch.pop('complete',None)
            if complete is not None and not complete(True):
                _fail_batch(target,'Scope acquisition was superseded',stale=True)
                return
            candidates=tuple(item['candidate'] for item in batch['members'])
            batch['protocol'].require('captured','publish',payload=id(candidates),validated=True)
            for item in batch['members']:
                result=item.pop('candidate')
                item['result']=result
                grid=result.get('layout') or result.get('sample_grid')
                if grid is None:
                    from ...perceptual_grade_core.scopes.samples import ordinary_layout
                    layout=ordinary_layout(result['rgba'].shape[1],result['rgba'].shape[0])
                    grid={name:getattr(layout,name) for name in ('mass','x','uv','tiles','ids','key')}
                item['request']['_sample_grid']={name:grid[name] for name in ('mass','x','uv','tiles','ids','key')}
                item['accepted']=dict(request=item['request'],result=result,size=item['size'],
                                      exact_size=item['exact_size'])
                item.update(stale=False,phase='ready',error='')
                _trace('capture_adopted',item['request'],attempt=item.get('_diagnostic_attempt',0),
                       domain=tuple(result.get('domain',())),samples=tuple(result['rgba'].shape[:2]),
                       helper_timings=result.get('_capture_timings',{}))
            batch.update(phase='ready',error='')
            _finish_batch_protocol(batch)
        except Exception as error:
            _fail_batch(target,str(error))
        finally:
            ownership.release(batch.pop('node',None))

    def acquire(node, inputs, *, complete=None):
        """Acquire one immutable automatic primary/reference source batch.

        Each input is (socket_name, descriptor, sample_size_or_None, exact_size).
        A reference with no size waits for the primary result's analysis grid.
        This function does not rediscover an editor or a newer source revision.
        """
        values=tuple(inputs)
        if not values:return False
        targets={_target(node,item[1]) for item in values}
        if len(targets)!=1:raise ValueError('Scope inputs belong to different editor occurrences')
        target=next(iter(targets));previous=batches.get(target)
        if previous is not None and previous['phase']=='capturing':return False
        if len({item[0] for item in values})!=len(values):raise ValueError('Duplicate scope input')
        for socket,descriptor,size,exact in values:
            if descriptor.get('_target_pointer')!=node.as_pointer() or descriptor.get('socket_name')!=socket:
                raise ValueError('Scope input descriptor belongs to another target')
        scheduler=application.component('graph.scheduler')
        request_id=next(input_requests)
        batch=dict(request=request_id,phase='capturing',error='',members=[],
                   node=ownership.retain(node,'scope.input.batch'),checked=0.,
                   prepare=None,release=None,complete=complete)
        batch_identity=dict(request=request_id,generation=input_generation,
                            source=scheduler.identity(('scope-input-batch',tuple(item[1]['identity'] for item in values))),
                            target=scheduler.identity(('scope-input-batch-target',target)))
        batch['protocol']=Protocol(application,'scope_input',initial={'generation':input_generation},identity=batch_identity)
        batch['protocol'].require('request','prepare')
        batches[target]=batch;batches.move_to_end(target)
        try:
            for socket,descriptor,size,exact in values:
                old=_input_state(node,socket,descriptor)
                accepted=None if old is None else old.get('accepted')
                for owner,item in tuple(states.items()):
                    if owner[:2]==target and owner[2]==socket:
                        states.pop(owner);_retire(item)
                descriptor=dict(descriptor)
                sample_size=tuple(size or (256,192))
                owner=target+(socket,sample_size)+(('exact',) if exact else ())
                identity=dict(request=request_id,generation=input_generation,
                              source=scheduler.identity(('scope-input',descriptor['identity'])),
                              target=scheduler.identity(('scope-input-target',target,socket)))
                authority=Protocol(application,'scope_input',initial={'generation':input_generation},identity=identity)
                authority.require('request','prepare')
                state=dict(request=descriptor,size=sample_size,exact_size=bool(exact),
                           primary_grid=bool(exact and size is None and socket!='Color'),
                           due=0.,node=ownership.retain(node,'scope.capture.node'),
                           handle=None,result=None,error='',directory=None,accepted=accepted,
                           stale=bool(accepted),phase='capturing',batch=target,input_protocol=authority,
                           defer_submit=bool(exact and size is None and socket!='Color'),_diagnostic_attempt=0)
                states[owner]=state;batch['members'].append(state)
                _trace('request_created',descriptor,sample_size=sample_size,exact_size=bool(exact))
            while len(batches)>MAX_CAPTURES:
                expired=next(iter(batches));_fail_batch(expired,'Scope input cache was retired',stale=True)
                old=batches.pop(expired)
                for owner,item in tuple(states.items()):
                    if owner[:2]==expired:states.pop(owner);_retire(item)
            for item in batch['members']:
                _snapshot(item)
            scheduler.wake(poll,.001)
        except Exception as error:
            _fail_batch(target,str(error))
            raise
        return True

    def accepted_descriptor(node, socket_name='Color', context=None):
        state=_input_state(node,socket_name,context)
        accepted=None if state is None else state.get('accepted')
        return None if accepted is None else dict(accepted['request'])

    def status(node, context=None):
        batch=batches.get(_target(node,context))
        if batch is None:return dict(state='waiting',pending=False,available=False,error='')
        available=any(item.get('accepted') is not None for item in batch['members'])
        phase=batch['phase']
        return dict(state='capturing' if phase=='capturing' else 'captured' if phase=='ready' else
                    'refresh_needed' if phase=='stale' else 'error',pending=phase=='capturing',
                    available=available,error=batch.get('error',''))

    def busy(node,context=None):
        if node is None:return False
        target=_target(node,context)
        batch=batches.get(target)
        if batch is not None and batch['phase']=='capturing':return True
        return any(item.get('batch')==target and item.get('handle') is not None
                   and not item['handle'].retired() for item in retiring)

    def cancel_scope(node, context=None):
        if node is None:return False
        target=_target(node,context)
        if target not in batches:return False
        _fail_batch(target,'Scope acquisition context changed',stale=True)
        return True

    def mark_stale(node, context=None, reason='Scope inputs changed'):
        target=_target(node,context);batch=batches.get(target)
        if batch is None or batch['phase'] in {'stale','error'}:return False
        _fail_batch(target,reason,stale=True)
        return True

    def fail(node, context, error):
        """Retain an acquisition's preparation error on its exact initiating target."""
        target=_target(node,context)
        if target not in batches:
            batches[target]=dict(phase='error',error=str(error),members=[],checked=0.)
        else:_fail_batch(target,str(error))

    def _watch_changed(request):
        scene=bpy.data.scenes.get(request['scene'])
        if scene is None:return True
        _poll_updates(scene,request['view'])
        if _scene_watch(scene,request.get('_watch_needs_scene',False))!=request.get('_watch_scene'):return True
        if _view_watch(request)!=request.get('_watch_view'):return True
        if _image_metadata(item[0] for item in request.get('_watch_images',()))!=request.get('_watch_images'):return True
        watched=request.get('_watch_nodes',())
        needed={key[0] for key,value in watched}
        trees={};incoming={}
        try:
            for tree in bpy.data.node_groups:
                if int(tree.session_uid) in needed:trees[int(tree.session_uid)]=tree
            for owners in (bpy.data.materials,bpy.data.worlds,bpy.data.lights):
                for owner in owners:
                    tree=getattr(owner,'node_tree',None)
                    if tree is not None and int(tree.session_uid) in needed:trees[int(tree.session_uid)]=tree
            def links(socket):
                uid=int(socket.id_data.session_uid)
                if uid not in incoming:
                    values={}
                    for link in socket.id_data.links:
                        values.setdefault(link.to_socket.as_pointer(),[]).append(link)
                    incoming[uid]=values
                return incoming[uid].get(socket.as_pointer(),())
            for (uid,name,pointer,output),before in watched:
                tree=trees.get(uid);node=tree.nodes.get(name) if tree is not None else None
                if node is None or node.as_pointer()!=pointer or _node_watch(node,output,links)!=before:return True
            return False
        finally:
            incoming.clear();trees.clear()

    def check_stale(node, context=None, force=False):
        """Bounded foreground observation of an authorized source's dependencies.

        No graph traversal, scene fingerprint, snapshot or helper admission is
        performed here. Call only for visible consumers, never from drawing.
        """
        target=_target(node,context);batch=batches.get(target)
        if batch is None or batch['phase'] not in {'ready','capturing'}:return False
        now=time.monotonic()
        if not force and now-batch['checked']<.5:return False
        if not application.component('graph.output_guard').available():return False
        batch['checked']=now
        try:changed=any(_watch_changed(item['request']) for item in batch['members'])
        except (ValueError,ReferenceError,RuntimeError,OSError):changed=True
        if changed:return mark_stale(node,context)
        return False

    def sample(node, socket_name='Color', size=None, descriptor=None,exact_size=False):
        """Retrieve accepted immutable pixels; queries never authorize capture."""
        state=_input_state(node,socket_name,descriptor)
        accepted=None if state is None else state.get('accepted')
        if accepted is not None:
            result=accepted['result']
            if descriptor is not None and descriptor['identity']!=result['identity']:
                raise application.component('graph.lut').ScopePreparing('Scope inputs changed; refresh again')
            if exact_size and size is not None and tuple(size)!=(result['rgba'].shape[1],result['rgba'].shape[0]):
                raise ValueError('Refresh the reference for the current primary sample grid')
            return result['rgba'],result['identity'],result.get('label','Native graph')
        if state is not None and state.get('error'):raise ValueError(state['error'])
        message='Evaluating Blender graph' if state is not None and state.get('phase')=='capturing' else 'Waiting for automatic scope inputs'
        raise application.component('graph.lut').ScopePreparing(message)

    def retry(node):
        """Numerical retries cannot grant a second native acquisition."""
        return False

    def _upstream_grades(target, location):
        """Follow connected outputs, including exact nested-group occurrences.

        Native PG Color outputs require current kernels; Baked Color depends
        only on its bound Image. Render Layers additionally reads scene shaders.
        """
        nodes = application.component('graph.nodes')
        stacks = application.component('graph.stacks')
        scene = bpy.data.scenes.get(location['scene'])
        if scene is None:raise ReferenceError('The compositor bake scene was removed')
        parents = []
        tree = scene.compositing_node_group
        for name in location.get('group_path', ()):
            instance = tree.nodes.get(name)
            if instance is None or instance.node_tree is None:
                raise ReferenceError('The compositor bake group occurrence was removed')
            parents.append(instance);tree = instance.node_tree
        target_parents = tuple(n.as_pointer() for n in parents)
        seen = set(); rendered = set(); required = OrderedDict()

        work = [('input', target.inputs['Color'], tuple(parents))]
        incoming = {}
        def input_links(socket):
            owner = socket.id_data; pointer = owner.as_pointer()
            if pointer not in incoming:
                indexed = {}
                for link in owner.links:indexed.setdefault(link.to_socket.as_pointer(), []).append(link)
                incoming[pointer] = indexed
            return incoming[pointer].get(socket.as_pointer(), ())
        try:
            while work:
                kind, value, path = work.pop()
                if kind == 'require':
                    required[value.as_pointer()] = value
                    continue
                if kind == 'input':
                    work.extend(('output', link.from_socket, path) for link in reversed(input_links(value))
                                if not link.is_muted and link.is_valid)
                    continue
                if kind == 'shaders':
                    if value.as_pointer() in rendered:continue
                    rendered.add(value.as_pointer())
                    trees = application.component('graph.output')._render_trees(value)
                    inputs = []
                    for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights):
                        for owner in collection:
                            root = getattr(owner, 'node_tree', None)
                            if root is None or root.as_pointer() not in trees:continue
                            for endpoint in root.nodes:
                                if endpoint.bl_idname not in {'ShaderNodeOutputMaterial', 'ShaderNodeOutputWorld', 'ShaderNodeOutputLight'}:
                                    continue
                                if not getattr(endpoint, 'is_active_output', True):continue
                                inputs.extend(('input', socket, ()) for socket in endpoint.inputs)
                    work.extend(reversed(inputs));inputs.clear()
                    continue
                socket = value; node = socket.node
                key = (node.as_pointer(), socket.identifier, tuple(item.as_pointer() for item in path))
                if key in seen:continue
                seen.add(key)
                if node.mute:
                    bypass = tuple(link.from_socket for link in node.internal_links if link.to_socket == socket)
                    if bypass:
                        work.extend(('input', source, path) for source in reversed(bypass))
                        continue
                if nodes.is_grade(node) or stacks.is_stack(node):
                    if socket.name in {'Baked Color','Baked Alpha'}:continue
                    if socket.name == 'Alpha':
                        if 'Alpha' in node.inputs:work.append(('input', node.inputs['Alpha'], path))
                        continue
                    if node == target and tuple(item.as_pointer() for item in path) == target_parents:
                        raise ValueError('Compositor bake input contains a grading cycle')
                    # Preserve postorder: prepare upstream kernels before their consumers.
                    work.append(('require', node, path))
                    if 'Color' in node.inputs:work.append(('input', node.inputs['Color'], path))
                    continue
                if node.bl_idname == 'NodeGroupInput':
                    if path:
                        source = next((item for item in path[-1].inputs if item.identifier == socket.identifier), None)
                        if source is not None:work.append(('input', source, path[:-1]))
                    continue
                group = getattr(node, 'node_tree', None)
                if group is not None:
                    for endpoint in reversed(tuple(group.nodes)):
                        if endpoint.bl_idname == 'NodeGroupOutput' and endpoint.is_active_output:
                            source = next((item for item in endpoint.inputs if item.identifier == socket.identifier), None)
                            if source is not None:work.append(('input', source, (*path, node)))
                    continue
                work.extend(('input', source, path) for source in reversed(tuple(node.inputs)))
                if node.bl_idname == 'CompositorNodeRLayers':
                    work.append(('shaders', getattr(node, 'scene', None) or scene, ()))
            return tuple(required.values())
        finally:
            work.clear()
            incoming.clear()
            required.clear()
            parents.clear()

    def _bake_grade_dependencies(target, location, shader=False):
        """Return the exact live grading outputs used by one Bake request."""
        values = ([*application.component('scopes.bake_routes').grade_dependencies(
                    target, location)] if shader else
                  [target, *_upstream_grades(target, location)])
        unique = []
        seen = set()
        for node in values:
            try:
                pointer = int(node.as_pointer())
            except (AttributeError, ReferenceError, RuntimeError):
                continue
            if pointer and pointer not in seen:
                seen.add(pointer)
                unique.append(node)
        return tuple(unique)

    def _bake_inputs_changed_state(state, native_lease=False, force=False):
        """Observe source-local revisions without walking downstream consumers."""
        if state.get('inputs_changed'):
            return True
        if not native_lease and not application.component('graph.output_guard').available():
            return False
        nodes = application.component('graph.nodes')
        for reference, revision in state.get('grade_inputs', ()):
            node = ownership.resolve(reference)
            if node is None:
                return True
            try:
                if nodes.observe_intent(node) != revision:
                    return True
            except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
                return True
        node = ownership.resolve(state.get('node'))
        if node is None:
            # Captured results keep an observation owner until publication or
            # cancellation even after their worker has physically retired.
            node = ownership.resolve(state.get('input_target'))
        if node is None:
            return True
        now = time.monotonic()
        if not force and now - state.get('input_checked', 0.0) < .2:
            return False
        state['input_checked'] = now
        try:
            if native_lease:
                return _bake_input_observation(node, state['input_context'], native_lease=True) != state['input_observation']
            with application.component('graph.output_guard').native_operation():
                return _bake_input_observation(node, state['input_context'], native_lease=True) != state['input_observation']
        except application.component('graph.output_guard').PublicationDeferred:
            return False
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError, OSError):
            return True

    def _bake_input_context(node, location):
        contexts = application.component('scopes.context')
        parents = contexts.occurrence(node, location)
        root = parents[0].id_data if parents else node.id_data
        scene = bpy.data.scenes.get(location.get('scene', ''))
        route = location.get('shader_bake', {})
        obj = bpy.data.objects.get(route.get('object_name', ''))
        return dict(_lightweight=True, root_pointer=int(root.as_pointer()),
                    tree_pointer=int(node.id_data.as_pointer()),
                    group_path=tuple(value.name for value in parents),
                    group_pointers=tuple(int(value.as_pointer()) for value in parents),
                    scene_uid=int(scene.session_uid) if scene else 0,
                    scene=scene.name if scene else '',
                    object_uid=int(obj.session_uid) if obj else 0,
                    domain=route.get('domain', 'COMPOSITOR'))

    def _bake_input_observation(node, context, *, native_lease=False):
        """Read this Bake's exact inputs, excluding generated output resources.

        This is a foreground, request-owned observation. Native callbacks only
        advance source-local semantic revisions and never enumerate consumers.
        """
        contexts = application.component('scopes.context')
        context = contexts.source_context(node, context)
        parents = contexts.occurrence(node, context)
        nodes = application.component('graph.nodes')
        intent = nodes.output_intent(node)
        scene = next((value for value in bpy.data.scenes
                      if int(value.session_uid) == context['scene_uid']), None)
        if context['scene_uid'] and scene is None:
            raise ReferenceError('The Bake scene was removed')
        manifest = {}; graphs = []; images = set()
        owner = {'scene': scene.name} if scene else None
        for name in ('Color', 'Alpha'):
            if name not in node.inputs:
                continue
            graph, names = _graph(node, name, parents=parents, manifest=manifest,
                                 owner=owner, bake_watch=True)
            graphs.append((name, graph));images.update(names)
        scene_graphs = []
        for referenced in manifest.get('scenes', {}).values():
            scene_manifest = {}
            graph, names = _scene_shaders(referenced, manifest=scene_manifest, bake_watch=True)
            scene_graphs.append((int(referenced.session_uid), graph,
                                 tuple(scene_manifest.get('structure', ()))))
            images.update(names)
        surface = None
        if context.get('object_uid'):
            obj = next((value for value in bpy.data.objects
                        if int(value.session_uid) == context['object_uid']), None)
            if obj is None:
                raise ReferenceError('The Bake mesh was removed')
            surface = (_properties(obj), _geometry_signature(obj, fresh=True),
                       tuple(tuple(row) for row in obj.matrix_world))
        frame = (scene.frame_current, scene.frame_subframe, scene.render.engine) if scene else None
        sizes = (int(node.get('_pg_bake_width', 2048)), int(node.get('_pg_bake_height', 2048)))
        return _fingerprint((intent, nodes.observe_intent(node, intent), tuple(graphs),
                             tuple(manifest.get('structure', ())),
                             _image_state(images, native_lease=native_lease), tuple(scene_graphs),
                             surface, frame, sizes))

    def _interrupt_bake_inputs(state):
        if state.get('inputs_changed'):
            return
        state['inputs_changed'] = True
        state['error'] = 'Bake inputs changed'
        state['result'] = None

    class BakeInputsChanged(RuntimeError):
        """The live Grade/Stack dependencies changed after Bake admission."""

    def check_bake_inputs(token, *, native_lease=False, force=False):
        state = bakes.get(token)
        if state is None:
            raise KeyError('Unknown compositor bake')
        if force:
            image_probes.clear()
        if _bake_inputs_changed_state(state, native_lease=native_lease, force=force):
            _interrupt_bake_inputs(state)
            raise BakeInputsChanged('Bake inputs changed')
        return True

    def _prepare_bake_input(target, location):
        nodes = application.component('graph.nodes')
        stacks = application.component('graph.stacks')
        lut = application.component('graph.lut')
        application.component('graph.output_guard').require_available()
        application.component('ownership.transactions').require_clean()
        if nodes.recovery_pending():
            raise RuntimeError('Grading data is still recovering; try the bake again shortly')
        rendering = nodes._rendering
        try:
            nodes._rendering = True
            for node in _upstream_grades(target, location):
                if lut.is_adopted(node) and not lut.request_intent(node, nodes.output_intent(node)):
                    continue
                if not nodes.resources_ready(node):
                    raise ValueError('An upstream Grade output is not ready')
                if stacks.is_stack(node):stacks.refresh(node)
                else:nodes.refresh_node(node)
                model = nodes.bake_model(node)
                lut.ensure(node, model, force=True)
                lut.validate_output(node, model, final=True)
        finally:
            nodes._rendering = rendering

    def _prepare_shader_uv_target(target):
        """Install the selected node's exact managed cube before surface baking."""
        nodes = application.component('graph.nodes')
        stacks = application.component('graph.stacks')
        lut = application.component('graph.lut')
        application.component('graph.output_guard').require_available()
        application.component('ownership.transactions').require_clean()
        if nodes.recovery_pending():
            raise RuntimeError('Grading data is still recovering; try the bake again shortly')
        if lut.is_adopted(target):
            return
        if not nodes.resources_ready(target):
            raise ValueError('The selected Grade output is not ready')
        rendering = nodes._rendering
        try:
            nodes._rendering = True
            if stacks.is_stack(target):
                stacks.refresh(target, True)
            else:
                nodes.refresh_node(target, force=True, immediate=True)
            model = nodes.bake_model(target)
            lut.ensure(target, model, force=True)
            lut.validate_output(target, model, final=True)
        finally:
            nodes._rendering = rendering

    def _describe_bake(node, context, *, _native_lease=False):
        from ...perceptual_grade_core.jobs.codec import freeze_model
        from ...perceptual_grade_core.model.raster import SCHEMA
        shader = getattr(getattr(node, 'id_data', None), 'bl_idname', '') == 'ShaderNodeTree'
        if shader:
            request = application.component('scopes.bake_routes').describe(
                node, context=context)
            if request is None:
                raise ValueError('Shader baking requires an eligible Grade or Stack node')
        else:
            request = describe(node, 'Color', context=context, _native_lease=_native_lease)
            if request is None:raise ValueError('Compositor baking requires a scene-owned node tree')
        model = application.component('graph.nodes').bake_model(node)
        rows = application.component('graph.lut').scope_rows(node, model)
        analytic = dict(version=SCHEMA,
                        kind='STACK' if application.component('graph.stacks').is_stack(node) else 'GRADE',
                        model=freeze_model(model),
                        rows=tuple(tuple(tuple(float(v) for v in row) for row in matrix) for matrix in rows))
        analytic['signature'] = hashlib.sha256(repr(analytic).encode('utf8')).hexdigest()
        request['analytic'] = analytic
        request['identity'] = (('analytic-shader-bake-v1' if shader else
                                'analytic-compositor-bake-v1'),
                               request['identity'], analytic['signature'])
        return request

    def start_bake(node, context=None, *, defer_submit=False):
        """Admit one current-frame compositor output bake.

        The returned token is intentionally opaque; callers poll status and
        release the artifact after packing it into a persistent Blender Image.
        """
        if len(bakes) >= MAX_BAKES:
            raise RuntimeError('Compositor bake queue is full (8 requests)')
        nodes = application.component('graph.nodes')
        stacks = application.component('graph.stacks')
        tree_type = getattr(getattr(node, 'id_data', None), 'bl_idname', '')
        if node is None or tree_type not in ('CompositorNodeTree', 'ShaderNodeTree'):
            raise ValueError('Baking requires a Shader or Compositor node')
        if not (nodes.is_grade(node) or stacks.is_stack(node)):
            raise ValueError('Bake is available only for Perceptual Grade and Grade Stack nodes')
        shader = tree_type == 'ShaderNodeTree'
        if shader:
            routes = application.component('scopes.bake_routes')
            location = routes.describe(node, context=context)
            if location is None:
                raise ValueError('Shader bake route is unavailable')
            routes.prepare(node, location)
            if location['shader_bake']['domain'] == 'UV':
                _prepare_shader_uv_target(node)
            location = routes.describe(node, context=context)
        else:
            location = application.component('scopes.context').describe(node, 'Color', context=context)
            if location is None:
                raise ValueError('Compositor baking requires a scene-owned node tree')
            _prepare_bake_input(node, location)
        request = _describe_bake(node, location)
        if request.get('scene_pending'):
            raise RuntimeError('The scene is still settling; try the bake again shortly')
        token = uuid.uuid4().hex
        input_context = _bake_input_context(node, location)
        with application.component('graph.output_guard').native_operation():
            input_observation = _bake_input_observation(node, input_context, native_lease=True)
        grade_inputs = []
        try:
            for value in _bake_grade_dependencies(node, location, shader):
                reference = ownership.retain(value, 'scope.capture.bake.grade_input')
                try:
                    revision = application.component('graph.nodes').observe_intent(value)
                except Exception:
                    ownership.release(reference)
                    raise
                grade_inputs.append((reference, revision))
            grade_inputs = tuple(grade_inputs)
        except Exception:
            for reference, _revision in grade_inputs:
                ownership.release(reference)
            raise
        state = dict(request=request, bake=True, due=time.monotonic(), node=ownership.retain(
            node, 'scope.capture.bake.node'), handle=None, result=None, error='',
            directory=None, cancelled=False, terminal=False, defer_submit=bool(defer_submit),
            worker_operation='shader_bake' if shader else 'compositor_bake',
            grade_inputs=grade_inputs, inputs_changed=False, input_context=input_context,
            input_observation=input_observation,
            input_target=ownership.retain(node, 'scope.capture.bake.input_target'))
        bakes[token] = state
        try:
            # Freeze the source before returning to Blender's event loop. The
            # user may edit the live graph while the helper evaluates this copy.
            _snapshot(state)
            if state.get('handle') is None and state.get('payload') is None:
                raise RuntimeError('Could not freeze a stable bake snapshot')
        except Exception:
            bakes.pop(token, None)
            ownership.release(state.pop('node',None))
            _dispose_bake_files(state)
            raise
        try:
            application.component('graph.scheduler').wake(poll, .05)
        except RuntimeError:
            # Capture admission already succeeded. Preserve its token so the
            # foreground can poll or cancel the exact physical attempt.
            logging.getLogger(__name__).exception('Could not schedule an accepted compositor bake')
        return token

    def dispatch_bake(token, lane):
        """Submit one already frozen click after the bake queue grants a lane."""
        if lane not in ('bake_0', 'bake_1'):
            raise ValueError('Unknown compositor bake worker lane')
        state = bakes[token]
        if (state.get('cancelled') or state.get('terminal') or state.get('handle') is not None
                or not state.get('defer_submit') or state.get('payload') is None):
            raise RuntimeError('Compositor bake snapshot cannot be dispatched again')
        runtime = application.component('services').get_runtime()
        operation = state.get('worker_operation', 'compositor_bake')
        state['handle'] = runtime.submit(JobSpec(operation, state['payload'], lane=lane, key=None))
        state.pop('payload')
        _trace_submit(runtime, state['request'], state.get('_diagnostic_attempt', 0), operation, lane)
        return True

    def bake_status(token):
        if token not in bakes:
            raise KeyError('Unknown compositor bake')
        poll()
        state = bakes[token]
        if state.get('inputs_changed'):
            raise BakeInputsChanged('Bake inputs changed')
        if state.get('error'):
            raise ValueError(state['error'])
        if state.get('cancelled'):
            raise ValueError('Compositor bake cancelled')
        if state.get('result') is not None:
            return dict(state['result'])
        return None

    def cancel_bake(token):
        """Retain the cancellation diagnostic until explicit consumer release."""
        state = bakes.get(token)
        if state is None:
            return False
        state['cancelled'] = True
        state['terminal'] = True
        state['error'] = state.get('error') or 'Compositor bake cancelled'
        state['result'] = None
        _retire_bake(state)
        if (bake_retiring and not application.closing
                and not application.app.fenced and not application.app.closed):
            application.component('graph.scheduler').wake(poll, .05)
        return True

    def on_bake_retired(token, callback):
        """Register ownership cleanup before release or native clearing can run."""
        state = bakes.get(token)
        if state is None:
            raise KeyError('Unknown compositor bake')
        state['on_retired'] = callback

    def release_bake(token, *, on_retired=None):
        """Drop consumer state; outstanding file leases retire independently."""
        state = bakes.pop(token, None)
        if state is None:
            return False
        if on_retired is not None:
            state['on_retired'] = on_retired
        _retire_bake(state)
        if (bake_retiring and not application.closing
                and not application.app.fenced and not application.app.closed):
            application.component('graph.scheduler').wake(poll, .05)
        return True

    def invalidate(scene=None, depsgraph=None):
        nonlocal update_epoch
        if depsgraph is None:
            # The native handler may run on another thread: only a Python scalar
            # changes here; no RNA or component discovery belongs to callbacks.
            update_epoch+=1
            return
        for update in depsgraph.updates:
            value = getattr(update.id,'original',update.id)
            if isinstance(value,bpy.types.Image):
                if value.get('_pg_cube_signature'):continue
                kind='image'
            elif isinstance(value,bpy.types.Object):
                if not (update.is_updated_geometry or update.is_updated_transform):continue
                kind='object'
            elif isinstance(value,(bpy.types.Mesh,bpy.types.Curve,bpy.types.Light,bpy.types.Camera)):kind='data'
            else:continue
            key=(kind,value.as_pointer())
            revisions[key]=revisions.get(key,0)+1

    def clear_scopes():
        """Fence scope authority without cancelling independent compositor Bakes."""
        nonlocal input_generation
        input_generation += 1
        for target in tuple(batches):_fail_batch(target,'Scope lifecycle changed',stale=True)
        batches.clear()
        for state in tuple(states.values()):
            _retire(state)
        states.clear()
        if retiring and not application.closing and not application.app.fenced and not application.app.closed:
            application.component('graph.scheduler').wake(poll,.05)

    def clear():
        nonlocal generation,update_epoch
        clear_scopes()
        generation += 1
        processed_epochs.clear();update_epoch=1
        revisions.clear();image_probes.clear();geometry_probes.clear()
        for token, state in tuple(bakes.items()):
            _retire_bake(state)
            bakes.pop(token, None)
        # Cancellation does not release input leases. A later poll or the
        # execution session's crash recovery reclaims physically retired inputs.
        if ((retiring or bake_retiring) and not application.closing
                and not application.app.fenced and not application.app.closed):
            application.component('graph.scheduler').wake(poll,.05)

    return Component(dict(describe=describe,describe_source=describe_source,channel=channel,sample=sample,poll=poll,
                          acquire=acquire,busy=busy,cancel_scope=cancel_scope,accepted_descriptor=accepted_descriptor,
                          status=status,mark_stale=mark_stale,check_stale=check_stale,fail=fail,clear_scopes=clear_scopes,
                          enable_diagnostics=enable_diagnostics,diagnostics=diagnostics,
                          collect_retired=collect_retired,
                          start_bake=start_bake,dispatch_bake=dispatch_bake,bake_status=bake_status,
                          check_bake_inputs=check_bake_inputs,
                          BakeInputsChanged=BakeInputsChanged,
                          bake_input_observation=_bake_input_observation,
                          cancel_bake=cancel_bake,release_bake=release_bake,
                          on_bake_retired=on_bake_retired,
                          retry=retry,invalidate=invalidate,clear=clear,
                          __name__=__name__), readers={'states':lambda:states,
                                                       'retiring':lambda:retiring,
                                                       'bakes':lambda:bakes,
                                                       'bake_retiring':lambda:bake_retiring})
