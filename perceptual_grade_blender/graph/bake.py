"""Explicit image-domain and UV baking for Shader and Compositor Grade nodes.

The numerical LUT baker and this graph-result baker are deliberately separate.
This component owns the foreground side of a native compositor capture: the
helper returns a verified EXR, Blender loads it into a float Image, and the
Image is packed before the helper lease is released.
"""

from ..integration.components import Component
from ..runtime.protocol import Protocol


def create(application):
    import bpy
    from bpy.app.handlers import persistent
    from bpy.props import PointerProperty
    from pathlib import Path
    from itertools import count
    import hashlib
    import json
    import uuid
    from contextlib import contextmanager
    from bpy.props import StringProperty
    from bpy.props import IntProperty
    import numpy as np

    i18n = application.component('localization.i18n')
    native = i18n.native
    tr = i18n.tr
    grades = application.component('graph.grades')
    nodes = application.component('graph.nodes')
    stacks = application.component('graph.stacks')
    capture = application.component('scopes.capture')
    scheduler = application.component('graph.scheduler')
    guard = application.component('graph.output_guard')
    ownership = application.component('ownership')

    SCHEMA = 1
    TAG = '_pg_bake_schema'
    KERNEL = '_pg_bake_kernel'
    OUTPUT = native('bake.output.label')
    BAKED_COLOR = 'Baked Color'
    BAKED_ALPHA = 'Baked Alpha'
    BAKED_VECTOR = 'Vector'
    SOURCE = '_pg_baked_image'
    OUTPUT_SCHEMA = '_pg_bake_output_schema'
    OPERATION = '_pg_bake_operation'
    _classes = []
    _states = {}
    _timers = {}
    _freeing = set()
    _binding_suppressed = set()
    _replacement_images = {}
    _fenced_states = {}
    _recovery_completed = 0
    _recovery_total = 0
    _recovery_outcomes = {}
    _recovery_observations = count(1)
    _operations = count(1)
    _queue = None
    _replacement_pending = False
    _last_error = ''
    _last_error_node = None
    _errors = {}
    _interruptions = set()
    _playback_pending = set()
    _registered = False
    _menu_registered = False
    _preflights = {}
    _preflight_generation = None
    _preflight_revision = 0

    def playback_active(context=None):
        """Playback in any window blocks explicit image baking globally."""
        screen = getattr(context, 'screen', None)
        return bool((screen is not None and getattr(screen, 'is_animation_playing', False))
                    or any(getattr(item, 'is_animation_playing', False)
                           for item in bpy.data.screens))

    @persistent
    def preflight_invalidated(*_args):
        """Native updates revoke cached observations without evaluating anything."""
        nonlocal _preflight_revision
        _preflight_revision += 1

    def _preflight_epoch():
        nonlocal _preflight_generation
        generation = ownership.generation_key()
        if generation != _preflight_generation:
            _preflights.clear()
            _preflight_generation = generation
        return generation

    def _preflight_identity(node, context):
        routes = application.component('scopes.bake_routes')
        observed, frozen = routes.observation_key(node, context)
        return (_preflight_epoch(), _preflight_revision, repr(observed)), frozen

    def _preflight_observation(node, context, observation):
        # Constant image-domain graphs have no Image content or UV geometry to
        # validate. Their complete eligibility is already known passively.
        if observation['route']['domain'] == 'IMAGE' and observation['route']['source'] == 'constant':
            return observation
        identity, frozen = _preflight_identity(node, context)
        key = (_node_key(node), repr(frozen))
        record = _preflights.get(key)
        if record is None or record['identity'] != identity:
            record = dict(identity=identity, context=frozen, pending=True, code='', descriptor=None)
            _preflights[key] = record
            while len(_preflights) > 64:
                _preflights.pop(next(iter(_preflights)))
        if record['pending']:
            return dict(state='waiting', code='checking', route=observation['route'])
        return dict(state='blocked' if record['code'] else 'ready',
                    code=record['code'], route=observation['route'])

    def preflight_tick():
        """Validate one queried target on a safe foreground turn, before admission."""
        _preflight_epoch()
        guard.reconcile()
        if playback_active() or not guard.available():
            return .1
        routes = application.component('scopes.bake_routes')
        for key, record in tuple(_preflights.items()):
            if not record['pending']:
                continue
            try:
                node = routes.resolve_context(record['context'])
                identity, _frozen = _preflight_identity(node, record['context'])
                if _node_key(node) != key[0] or identity != record['identity']:
                    _preflights.pop(key, None)
                    continue
                descriptor, code = None, ''
                try:
                    destination = baked_image(node)
                    if destination is not None and not destination.is_float:
                        raise routes.RouteBlocked('destination', 'Bake destination is not a float Image')
                    descriptor = routes.describe(node, record['context'])
                except routes.RouteBlocked as error:
                    code = error.code
                except Exception as error:
                    application.component('integration.outcomes').failure(
                        node, 'bake', error, recovery='perceptual_grade.check_bake_inputs')
                    code = 'unavailable'
                after, _frozen = _preflight_identity(node, record['context'])
                if (guard.available() and after == identity and _preflights.get(key) is record):
                    record.update(pending=False, code=code, descriptor=descriptor)
                else:
                    _preflights.pop(key, None)
            except (ReferenceError, RuntimeError, ValueError, TypeError, AttributeError):
                _preflights.pop(key, None)
            redraw()
            break
        return .1

    @persistent
    def playback_started(*_args):
        """Latch native playback entry; never cancel/mutate inside its callback."""
        _playback_pending.update(token for token, state in _states.items()
                                 if not state.get('committed'))
        if _playback_pending:
            scheduler.wake(poll, .0)

    def _cancel_for_playback(context=None):
        active = playback_active(context)
        if active:
            _playback_pending.update(token for token, state in _states.items()
                                     if not state.get('committed'))
        interrupted = bool(_playback_pending)
        for token in tuple(_playback_pending):
            state = _states.get(token)
            if state is None or state.get('committed'):
                _playback_pending.discard(token)
                continue
            _interruptions.add(state['target_key'])
            _error(state, tr('bake.interrupted'))
            try:
                cancel(token)
            except guard.PublicationDeferred:
                # An interrupted publication repair remains owned and retries
                # once native readers have finished. Its artifact cannot win.
                continue
            _playback_pending.discard(token)
        if interrupted:
            redraw()
        return active or interrupted

    def _inputs_current(state, *, native_lease=False, force=False):
        """Each request observes its own frozen inputs; producers never cancel it."""
        if state.get('committed'):
            return True
        if state.get('interruption') == 'bake.inputs_changed':
            return False
        node = ownership.resolve(state.get('target'))
        if node is None:
            return False
        try:
            if nodes.observe_intent(node) != state['input_revision']:
                return False
            checker = getattr(capture, 'check_bake_inputs', None)
            if checker is not None and state.get('token'):
                checker(state['token'], native_lease=native_lease, force=force)
            return True
        except Exception as error:
            changed = getattr(capture, 'BakeInputsChanged', ())
            if changed and isinstance(error, changed):
                return False
            if isinstance(error, (ReferenceError, KeyError)):
                return False
            raise

    def _interrupt_inputs(token, state):
        state['interruption'] = 'bake.inputs_changed'
        _error(state, tr('bake.inputs_changed'))
        try:
            cancel(token)
        except guard.PublicationDeferred:
            # A publication rollback stays owned until its native lease clears.
            pass
        redraw()

    def _cancel_for_inputs():
        if not guard.available():
            return
        for token, state in tuple(_states.items()):
            if not state.get('committed') and not _inputs_current(state):
                _interrupt_inputs(token, state)

    def _is_compositor(node):
        return bool(node and getattr(getattr(node, 'id_data', None), 'bl_idname', '')
                    == 'CompositorNodeTree')

    def _is_shader(node):
        return bool(node and getattr(getattr(node, 'id_data', None), 'bl_idname', '')
                    == 'ShaderNodeTree')

    def _is_tree(kernel, tree_type):
        return bool(kernel and getattr(kernel, 'bl_idname', '') == tree_type)

    def _editable_tree(tree):
        return bool(tree and tree.bl_idname in ('CompositorNodeTree', 'ShaderNodeTree')
                    and not tree.is_evaluated and grades.local(tree))

    def _target(context):
        node = (getattr(context, 'pg_node', None) or
                getattr(context, 'pg_stack_node', None))
        if node is None:
            space = getattr(context, 'space_data', None)
            tree = getattr(space, 'edit_tree', None) if space else None
            node = getattr(getattr(tree, 'nodes', None), 'active', None)
        return node

    def _valid(node):
        return bool((_is_compositor(node) or _is_shader(node)) and not node.id_data.is_evaluated
                    and grades.local(node.id_data)
                    and (nodes.is_grade(node) or stacks.is_stack(node)))

    def _unique_name(base):
        name = str(base)[:54] or 'Bake'
        if bpy.data.images.get(name) is None:
            return name
        index = 2
        while bpy.data.images.get(f'{name}.{index:03d}') is not None:
            index += 1
        return f'{name}.{index:03d}'

    def _pack(image):
        image.pack()
        packed = image.packed_file
        if packed is None or packed.size <= 0 or image.is_dirty:
            raise RuntimeError('Bake image packing did not complete')

    def _sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _image_source(kernel):
        if kernel is None:
            return None
        return next((item for item in kernel.nodes
                     if item.get(SOURCE) and item.bl_idname in
                     ('CompositorNodeImage', 'ShaderNodeTexImage')), None)

    def create_output(kernel):
        """Construct baked storage once, while creating a new private kernel."""
        if kernel is None or kernel.bl_idname not in ('CompositorNodeTree', 'ShaderNodeTree'):
            raise ValueError('Baked storage requires a Shader or Compositor kernel')
        if _image_source(kernel) is not None or OUTPUT_SCHEMA in kernel:
            raise ValueError('Baked storage is already constructed')
        shader = kernel.bl_idname == 'ShaderNodeTree'
        source = _image_source(kernel)
        if shader:
            vector = next((item for item in kernel.interface.items_tree
                           if item.item_type == 'SOCKET' and item.in_out == 'INPUT'
                           and item.name == BAKED_VECTOR), None)
            if vector is None:
                vector = kernel.interface.new_socket(name=BAKED_VECTOR, in_out='INPUT',
                                                     socket_type='NodeSocketVector')
                vector.default_value = (0.0, 0.0, 0.0)
            vector.description = native('bake.vector.tooltip')
            fields = []
            for name, socket_type in ((BAKED_COLOR, 'NodeSocketColor'),
                                      (BAKED_ALPHA, 'NodeSocketFloat')):
                field = next((item for item in kernel.interface.items_tree
                              if item.item_type == 'SOCKET' and item.in_out == 'OUTPUT'
                              and item.name == name), None)
                if field is None:
                    field = kernel.interface.new_socket(name=name, in_out='OUTPUT',
                                                        socket_type=socket_type)
                fields.append(field)
        else:
            field = next((item for item in kernel.interface.items_tree
                          if item.item_type == 'SOCKET' and item.in_out == 'OUTPUT'
                          and item.name == OUTPUT), None)
            if field is None:
                field = kernel.interface.new_socket(name=OUTPUT, in_out='OUTPUT', socket_type='NodeSocketColor')
        if source is None:
            source = kernel.nodes.new('ShaderNodeTexImage' if shader else 'CompositorNodeImage')
            source.name = 'Perceptual Grade Baked Image'
            source[SOURCE] = 1
        output = next((item for item in kernel.nodes
                       if item.bl_idname == 'NodeGroupOutput' and item.is_active_output), None)
        if output is None:
            output = kernel.nodes.new('NodeGroupOutput'); output.is_active_output = True
        if shader:
            group_input = next((item for item in kernel.nodes if item.bl_idname == 'NodeGroupInput'), None)
            if group_input is None: group_input = kernel.nodes.new('NodeGroupInput')
            texcoord = next((item for item in kernel.nodes if item.bl_idname == 'ShaderNodeTexCoord'
                             and item.get('_pg_bake_uv_fallback')), None)
            if texcoord is None:
                texcoord = kernel.nodes.new('ShaderNodeTexCoord'); texcoord['_pg_bake_uv_fallback'] = 1
            if not any(link.to_socket == source.inputs['Vector'] for link in kernel.links):
                kernel.links.new(texcoord.outputs['UV'], source.inputs['Vector'])
            for field, output_name in zip(fields, (BAKED_COLOR, BAKED_ALPHA)):
                socket = output.inputs[field.identifier]
                output_socket = source.outputs['Color' if output_name == BAKED_COLOR else 'Alpha']
                if not any(link.from_node == source for link in socket.links):
                    kernel.links.new(output_socket, socket)
        else:
            socket = output.inputs[field.identifier]
            if not any(link.from_node == source for link in socket.links):
                kernel.links.new(source.outputs['Image'], socket)
        kernel[OUTPUT_SCHEMA] = 1
        return source

    def output_current(kernel):
        if kernel is None or kernel.bl_idname not in ('CompositorNodeTree', 'ShaderNodeTree'):
            return False
        if _image_source(kernel) is None or kernel.get(OUTPUT_SCHEMA) != 1:
            return False
        if kernel.bl_idname == 'CompositorNodeTree':
            return any(item.item_type == 'SOCKET' and item.in_out == 'OUTPUT'
                       and item.name == OUTPUT for item in kernel.interface.items_tree)
        names = {(item.name, item.in_out) for item in kernel.interface.items_tree
                 if item.item_type == 'SOCKET'}
        return {(BAKED_VECTOR, 'INPUT'), (BAKED_COLOR, 'OUTPUT'),
                (BAKED_ALPHA, 'OUTPUT')} <= names

    def baked_image(node):
        """The saved selector pointer is the sole destination authority."""
        return getattr(node, 'pg_bake_image', None)

    def sync_vector(node):
        """Route a shader reader's explicit Vector or UV fallback on the main thread."""
        if not _is_shader(node):
            return
        kernel = getattr(node, 'node_tree', None)
        source = _image_source(kernel)
        if kernel is None or source is None:
            return
        group_input = next((item for item in kernel.nodes if item.bl_idname == 'NodeGroupInput'), None)
        texcoord = next((item for item in kernel.nodes if item.get('_pg_bake_uv_fallback')), None)
        if group_input is None or texcoord is None:
            return
        # Saved readers and Grades can be native groups after load/Undo.
        group_name = BAKED_VECTOR
        vector = node.inputs.get(group_name)
        if vector is None:
            return
        desired = group_input.outputs.get(group_name) if vector.is_linked else texcoord.outputs.get('UV')
        if desired is None:
            return
        for link in tuple(source.inputs['Vector'].links):
            if link.from_socket != desired:
                kernel.links.remove(link)
        if not any(link.from_socket == desired for link in source.inputs['Vector'].links):
            kernel.links.new(desired, source.inputs['Vector'])

    def sync_coordinates():
        """Synchronize shader reader vectors after native link edits."""
        trees = list(bpy.data.node_groups)
        trees.extend(item.node_tree for item in bpy.data.materials if item.node_tree)
        trees.extend(item.node_tree for item in bpy.data.worlds if item.node_tree)
        trees.extend(item.node_tree for item in bpy.data.lights if item.node_tree)
        seen = set()
        for tree in trees:
            pointer = tree.as_pointer()
            if pointer in seen or not _editable_tree(tree):
                continue
            seen.add(pointer)
            for node in tree.nodes:
                kernel = getattr(node, 'node_tree', None)
                if kernel is not None and _is_shader(node):
                    managed = (kernel.get(KERNEL) == SCHEMA or nodes.is_grade(node)
                               or stacks.is_stack(node))
                    if managed and grades.local(kernel) and kernel.users - int(kernel.use_fake_user) <= 1:
                        sync_vector(node)

    def is_baked_image(image):
        """Whether an Image is a current local packed managed destination."""
        if image is None:
            return False
        try:
            return (int(image.get(TAG, 0)) == 2
                    and image.get('_pg_bake_kind') in {'EMPTY', 'COMPOSITOR', 'SHADER_IMAGE', 'SHADER_UV'}
                    and not image.is_evaluated and grades.local(image)
                    and image.get('_pg_bake_storage') == 'FLOAT_RGBA'
                    and image.packed_file is not None
                    and image.packed_file.size > 0)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            return False

    def _image_allowed(node, image):
        if image is None:
            return True
        if not is_baked_image(image):
            return False
        kind = image.get('_pg_bake_kind')
        if _is_compositor(node):
            return kind in {'EMPTY', 'COMPOSITOR', 'SHADER_IMAGE'}
        return _is_shader(node) and kind in {'EMPTY', 'COMPOSITOR', 'SHADER_IMAGE', 'SHADER_UV'}

    @contextmanager
    def binding_suppressed(node):
        pointer = node.as_pointer()
        already = pointer in _binding_suppressed
        _binding_suppressed.add(pointer)
        try:
            yield
        finally:
            if not already:
                _binding_suppressed.discard(pointer)

    def materialize_binding(node, kernel=None):
        """Derive an existing native source from the canonical saved pointer."""
        kernel = node.node_tree if kernel is None else kernel
        source = _image_source(kernel)
        if source is None:
            raise ValueError('Baked output resources are unavailable')
        image = baked_image(node)
        source.image = image
        if _is_shader(node) and image is not None:
            interpolation = image.get('_pg_bake_interpolation')
            extension = image.get('_pg_bake_extension')
            if interpolation in {item.identifier for item in source.bl_rna.properties['interpolation'].enum_items}:
                source.interpolation = interpolation
            if extension in {item.identifier for item in source.bl_rna.properties['extension'].enum_items}:
                source.extension = extension
        kernel.update_tag()
        node.id_data.update_tag()

    def _assign_binding(node, image):
        with binding_suppressed(node):
            node.pg_bake_image = image
        materialize_binding(node)

    def _binding_objects(record, after=False):
        transactions = application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        node = None if tree is None else transactions.find_node(tree, record['node'])
        if node is None:
            raise ReferenceError('Image binding target is unavailable')
        kernel = transactions.find_tree(record.get('after_kernel') or record['kernel'] if after else record['kernel'])
        if kernel is None or node.node_tree != kernel:
            raise ReferenceError('Image binding kernel changed')
        identity = record['after_image' if after else 'before_image']
        image = transactions.find_id(identity, bpy.data.images)
        if identity is not None and image is None:
            raise ReferenceError('Image binding destination is unavailable')
        return node, image

    def verify_binding(record, after=False):
        node, image = _binding_objects(record, after)
        source = _image_source(node.node_tree)
        if baked_image(node) != image or source is None or source.image != image:
            raise RuntimeError('Image binding does not match its ownership receipt')

    def repair_binding(record):
        transactions = application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        target = None if tree is None else transactions.find_node(tree, record['node'])
        original = transactions.find_tree(record['kernel'])
        if target is None or original is None:
            raise ReferenceError('Image binding owner is unavailable')
        if record.get('after_kernel') is not None:
            with guard.publication():
                target.node_tree = original
        node, image = _binding_objects(record)
        with guard.publication():
            _assign_binding(node, image)
            source = _image_source(node.node_tree)
            for key in ('interpolation', 'extension'):
                if record.get(key) is not None:
                    setattr(source, key, record[key])
        verify_binding(record)
        _discard_binding_kernel(record)

    def _discard_binding_kernel(record):
        identity = record.get('after_kernel')
        if identity is None:
            return
        transactions = application.component('ownership.transactions')
        kernel = transactions.find_tree(identity)
        if kernel is not None:
            if kernel.users - int(kernel.use_fake_user) != 0:
                raise RuntimeError('Detached Image kernel still has native users')
            with guard.publication():
                application.app.blender.native.remove_node_group(kernel)

    def set_baked_image(node, image):
        """Apply both destination references as one reversible native edit."""
        if not _editable_tree(node.id_data) or not _image_allowed(node, image):
            raise ValueError('Bake destination must be a packed Perceptual Grade Image')
        kernel = node.node_tree
        reader = getattr(node, 'bl_idname', '') in {'PGShaderBake', 'PGCompositorBake'}
        detach = bool(reader and (node.get('_pg_bake_pending_copy')
                                  or kernel is not None and kernel.users - int(kernel.use_fake_user) > 1))
        if (kernel is None or not grades.local(kernel)
                or not reader and kernel.users - int(kernel.use_fake_user) > 1):
            raise ValueError('Make the node output resources independent before binding a bake')
        source = _image_source(kernel)
        if source is None:
            raise ValueError('Baked output resources are unavailable')
        if guard.publishing() and not detach:
            # The enclosing bake publication already owns its durable rollback.
            _assign_binding(node, image)
            return
        transactions = application.component('ownership.transactions')
        transactions.require_clean()
        record = dict(node=transactions.node_key(node), kernel=transactions.id_key(kernel),
                      after_kernel=None,
                      before_image=transactions.id_key(baked_image(node)),
                      after_image=transactions.id_key(image),
                      interpolation=getattr(source, 'interpolation', None),
                      extension=getattr(source, 'extension', None))
        tx = transactions.Transaction('bake_binding', record)
        record = tx.record
        try:
            with guard.publication():
                if detach:
                    def allocate():
                        copied = kernel.copy()
                        record['after_kernel'] = transactions.id_key(copied)
                        return copied
                    copied = tx.apply('binding_kernel_allocate', allocate,
                        lambda: _discard_binding_kernel(record), retirement=True)
                    tx.apply('binding_kernel_assign', lambda: setattr(node, 'node_tree', copied),
                        lambda: setattr(node, 'node_tree', kernel),
                        before_edges=(record['kernel'],), after_edges=(record['after_kernel'],))
                tx.apply('image_binding', lambda: _assign_binding(node, image),
                         lambda: repair_binding(record),
                         before_edges=(record['before_image'],) * 2,
                         after_edges=(record['after_image'],) * 2)
                verify_binding(record, after=True)
                tx.commit()
                if '_pg_bake_pending_copy' in node:
                    del node['_pg_bake_pending_copy']
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()

    def image_changed(node, context=None):
        if (node.id_data.is_evaluated or node.as_pointer() in _binding_suppressed
                or node.as_pointer() in _freeing or nodes.recovery_pending()):
            return
        candidate = baked_image(node)
        source = _image_source(getattr(node, 'node_tree', None))
        previous = None if source is None else source.image
        with binding_suppressed(node):
            node.pg_bake_image = previous
        outcomes = application.component('integration.outcomes')
        if not picker_ready(node, context):
            return
        if not _image_allowed(node, candidate):
            outcomes.set(node, 'image', 'bake.blocked.destination')
            return
        try:
            set_baked_image(node, candidate)
            outcomes.clear(node, 'image')
        except guard.PublicationDeferred:
            return
        except Exception as error:
            outcomes.failure(node, 'image', error)

    def image_property():
        return PointerProperty(type=bpy.types.Image, name=native('bake.destination.label'),
            description=native('bake.destination.tooltip'), translation_context=i18n.CONTEXT,
            poll=_image_allowed, update=image_changed, options=set())

    def _creation_target(record):
        transactions = application.component('ownership.transactions')
        tree = transactions.find_tree(record['node']['tree'])
        node = None if tree is None else transactions.find_node(tree, record['node'])
        if node is None:
            raise ReferenceError('New Image target is unavailable')
        return node

    def _created_image(record):
        transactions = application.component('ownership.transactions')
        if record.get('created_image') is not None:
            return transactions.find_id(record['created_image'], bpy.data.images)
        return bpy.data.images.get(record['name'])

    def repair_image_creation(record):
        node = _creation_target(record)
        image = _created_image(record)
        with guard.publication():
            if baked_image(node) == image:
                _assign_binding(node, None)
            if image is not None:
                _discard_image(image)
        if _created_image(record) is not None:
            raise RuntimeError('New Image allocation still has native users')

    def verify_image_creation(record, after=False):
        node = _creation_target(record)
        image = _created_image(record)
        if not after:
            if image is not None:
                raise RuntimeError('New Image allocation was not retired')
            return
        source = _image_source(node.node_tree)
        if (image is None or not is_baked_image(image) or image.get('_pg_bake_kind') != 'EMPTY'
                or baked_image(node) != image or source is None or source.image != image):
            raise RuntimeError('New Image does not match its ownership receipt')

    def new_image(node):
        """Create an empty packed destination; the first bake establishes its domain."""
        if baked_image(node) is not None:
            return None
        guard.reconcile()
        if not picker_ready(node):
            return None
        transactions = application.component('ownership.transactions')
        transactions.require_clean()
        if (getattr(node, 'bl_idname', '') in {'PGShaderBake', 'PGCompositorBake'}
                and (node.get('_pg_bake_pending_copy')
                     or node.node_tree.users - int(node.node_tree.use_fake_user) > 1)):
            set_baked_image(node, None)
        tx = transactions.Transaction('bake_create', dict(node=transactions.node_key(node),
            created_image=None, name=_unique_name('Image')))
        image = None
        def allocate():
            value = bpy.data.images.new(tx.record['name'], width=1, height=1,
                                        alpha=True, float_buffer=True, is_data=True)
            tx.record['created_image'] = transactions.id_key(value)
            return value
        try:
            with guard.publication():
                image = tx.apply('image_allocate', allocate,
                    lambda: repair_image_creation(tx.record), retirement=True)
                def initialize():
                    image.alpha_mode = 'CHANNEL_PACKED'
                    image.pixels.foreach_set((0., 0., 0., 0.))
                    image.update()
                    image[TAG] = 2
                    image['_pg_bake_storage'] = 'FLOAT_RGBA'
                    image['_pg_bake_kind'] = 'EMPTY'
                    _pack(image)
                    _write_operation(image, uuid.uuid4().hex, 'COMPLETE')
                tx.apply('image_initialize', initialize, lambda: None)
                tx.apply('image_bind', lambda: _assign_binding(node, image),
                    lambda: repair_image_creation(tx.record),
                    after_edges=(tx.record['created_image'],) * 2)
                verify_image_creation(tx.record, after=True)
                tx.commit()
            return image
        except Exception:
            tx.rollback()
            raise
        finally:
            tx.close()

    def _dimension_get(key):
        def get(node):
            return int(node.get(key, 2048))
        return get

    def _dimension_set(key):
        def set_value(node, value):
            node[key] = int(value)
        return set_value

    def _node_key(node):
        try:
            return (ownership.generation_key(), int(node.id_data.session_uid),
                    int(node.as_pointer()), node.get('_pg_ownership_identity'))
        except (AttributeError, ReferenceError, RuntimeError):
            return None

    def _active_state(node):
        key = _node_key(node)
        if not key:
            return None
        for state in _states.values():
            try:
                target = ownership.resolve(state.get('target'))
                if target is not None and _node_key(target) == key:
                    return state
            except (ReferenceError, RuntimeError):
                continue
        return None

    def _pool():
        nonlocal _queue
        if _queue is None or _queue.store.ref.closed or _queue.state.closing:
            # Each native replacement gets a fresh actor. Actor envelopes and
            # monotonically increasing operation IDs reject old-pool receipts.
            _queue = Protocol(application, 'bake_queue', identity={'generation': 1})
        return _queue

    def _queue_event(state, kind, **values):
        return state['queue'].event(kind, **state['reservation'], **values)

    def _queue_require(state, kind, action, **values):
        effect = _queue_event(state, kind, **values)
        if not effect.accepted or effect.action != action:
            raise RuntimeError('Bake admission rejected ' + kind)
        return effect

    def _release_reservation(state):
        pool = state.get('queue')
        if pool is None or pool.store.ref.closed:
            return
        reservation = state['reservation']
        entry = pool.state.entries[reservation['slot']]
        if entry.request != reservation['request']:
            return
        if entry.physical:
            _queue_require(state, 'retired', 'retire', validated=True)
        _queue_require(state, 'finish', 'release', validated=True)
        if pool.state.closing and not any(entry.request for entry in pool.state.entries):
            pool.close()

    def _error(state, error):
        nonlocal _last_error, _last_error_node
        _last_error = str(error)
        _last_error_node = state['target_key']
        _errors[_last_error_node] = _last_error
        state['error'] = _last_error
        node = ownership.resolve(state.get('target'))
        if node is not None:
            outcomes = application.component('integration.outcomes')
            if hasattr(error, 'code'):
                outcomes.set(node, 'bake', 'bake.blocked.' + error.code)
            elif state.get('interruption'):
                outcomes.set(node, 'bake', state['interruption'])
            elif state['target_key'] in _interruptions:
                outcomes.set(node, 'bake', 'bake.interrupted')
            elif isinstance(error, Exception):
                outcomes.failure(node, 'bake', error, recovery='perceptual_grade.bake_compositor')
            else:
                outcomes.set(node, 'bake', 'bake.blocked.failed', recovery='perceptual_grade.bake_compositor')

    def _dispatch():
        """The pool grants a free worker only to a frozen destination head."""
        if playback_active() or _playback_pending:
            return
        for state in tuple(_states.values()):
            if state.get('error') or state.get('committed'):
                continue
            entry = state['queue'].state.entries[state['reservation']['slot']]
            if entry.phase != 'queued':
                continue
            if not _inputs_current(state):
                _interrupt_inputs(state['token'], state)
                continue
            for lane in (1, 2):
                effect = _queue_event(state, 'start', lane=lane, validated=True)
                if not effect.accepted:
                    continue
                try:
                    capture.dispatch_bake(state['token'], 'bake_' + str(lane - 1))
                except Exception as error:
                    # Submission failure means no worker owns this payload.
                    _queue_event(state, 'cancel')
                    _queue_require(state, 'retired', 'retire', validated=True)
                    _error(state, error)
                break

    def _metadata(image, artifact):
        image[TAG] = 2
        image['_pg_bake_storage'] = 'FLOAT_RGBA'
        domain = str(artifact.get('bake_domain', 'COMPOSITOR'))
        image['_pg_bake_kind'] = {'IMAGE': 'SHADER_IMAGE', 'UV': 'SHADER_UV'}.get(domain, 'COMPOSITOR')
        image['_pg_bake_evaluation'] = str(artifact.get('evaluation', 'ANALYTIC'))
        image['_pg_bake_analytic_signature'] = str(artifact.get('analytic_signature', ''))
        image['_pg_bake_model_kind'] = str(artifact.get('analytic_kind', ''))
        image['_pg_bake_scene'] = str(artifact.get('scene', ''))
        image['_pg_bake_node'] = str(artifact.get('node', ''))
        image['_pg_bake_frame'] = int(artifact.get('frame', 0))
        image['_pg_bake_subframe'] = float(artifact.get('frame_subframe', 0.0))
        image['_pg_bake_engine'] = str(artifact.get('engine', ''))
        image['_pg_bake_width'] = int(artifact['width'])
        image['_pg_bake_height'] = int(artifact['height'])
        image['_pg_bake_domain'] = tuple(int(v) for v in artifact.get(
            'domain', (artifact['width'], artifact['height'])))
        image['_pg_bake_identity'] = repr(artifact.get('identity', ()))
        image['_pg_bake_sha256'] = str(artifact.get('sha256', ''))
        if artifact.get('bake_domain'):
            image['_pg_bake_object'] = str(artifact.get('object', ''))
            image['_pg_bake_uv_map'] = str(artifact.get('uv_map', ''))
            image['_pg_bake_interpolation'] = str(artifact.get('interpolation', ''))
            image['_pg_bake_extension'] = str(artifact.get('extension', ''))
        if '_pg_bake_pixels_sha256' in image:
            del image['_pg_bake_pixels_sha256']

    def _operation(image):
        try:
            value = json.loads(image.get(OPERATION, ''))
        except (TypeError, ValueError):
            return None
        return value if (isinstance(value, dict) and value.get('version') == 1
            and isinstance(value.get('id'), str) and len(value['id']) == 32
            and value.get('phase') in ('WRITING', 'COMPLETE')) else None

    def _commit(image):
        operation = _operation(image)
        if operation is not None and operation['phase'] == 'COMPLETE':
            return ('bake', operation['id'])
        # Permanent LUT producers use the same native-cache reconciliation.
        # A current producer receipt must identify its completed operation.
        try:
            operation = json.loads(image.get('_pg_payload_record', ''))
        except (TypeError, ValueError):
            return None
        if (isinstance(operation, dict) and operation.get('phase') == 'COMPLETE'
                and isinstance(operation.get('operation_id'), str)
                and len(operation['operation_id']) == 32):
            return ('lut', operation['operation_id'])
        return None

    def _write_operation(image, operation, phase):
        # This is historical producer completion, never current pixel validity.
        image[OPERATION] = json.dumps(dict(version=1, id=operation, phase=phase),
                                     sort_keys=True, separators=(',', ':'))

    def _write_packed(image, payload):
        """Complete a native packed transfer without inspecting its output."""
        image.pack(data=payload, data_len=len(payload))
        if image.packed_file is None or image.packed_file.size <= 0:
            raise RuntimeError('Bake packed-byte replacement failed')
        image.buffers_free()

    def _checkpoint(phase):
        """Fault-injection seam around native Image publication."""

    def _restore_native(state):
        repair = state.get('repair')
        if repair is None:
            return
        image = ownership.resolve(repair['image'])
        if image is None:
            raise ReferenceError('Baked Image repair lost its original Image')
        with guard.publication():
            if repair['kind'] == 'overwrite':
                _write_packed(image, repair['bytes'])
                for key in tuple(image.keys()):
                    if key.startswith('_pg_bake_'):
                        del image[key]
                for key, value in repair['metadata'].items():
                    image[key] = value
                image.update()
                image.update_tag()
            else:
                node = ownership.resolve(state['target'])
                kernel = ownership.resolve(state['kernel'])
                if node is None or kernel is None or node.node_tree != kernel:
                    raise ReferenceError('Baked output repair lost its original target')
                source = _image_source(kernel)
                if source is not None and source.image is not None and source.image != image:
                    raise ReferenceError('Baked output repair binding changed')
                _assign_binding(node, None)
                kernel.update_tag(); node.id_data.update_tag()

    def _discard_repair(state, *, restored=False):
        repair = state.get('repair')
        if repair is None:
            return
        image = ownership.resolve(repair['image']) if restored and repair['kind'] == 'attach' else None
        ownership.release(repair['image'])
        state.pop('repair')
        if image is not None:
            _discard_image(image)

    def _restore(state):
        if state.get('repair') is None:
            return
        if state.get('committed'):
            raise RuntimeError('A committed bake cannot be rolled back')
        publication = state.get('publication')
        if not state.get('mutation_started'):
            # An admission failure did not alter the target. The transaction's
            # backup is only bookkeeping until the first native mutation.
            if publication is not None and publication.state.debt:
                publication.require('aborted', 'release', validated=True)
            _discard_repair(state, restored=True)
            return
        if publication is not None:
            publication.event('failed')
            publication.require('repair', 'repair')
        try:
            _restore_native(state)
        except Exception:
            if publication is not None:
                publication.event('failed')
            raise
        if publication is not None:
            publication.require('repaired', 'release', validated=True, readable=True)
        # Keep the native backup through acknowledgement. A rejected receipt
        # must remain retryable even if restoring the pixels already succeeded.
        _discard_repair(state, restored=True)

    def _retirement(state):
        """Keep protocol actors until the capture adapter confirms retirement."""
        if not state.get('committed') and state.get('queue') is not None:
            _queue_event(state, 'cancel')
        workflow, publication = state.get('workflow'), state.get('publication')
        if workflow is not None and not workflow.store.ref.closed:
            workflow.event('close')
        if publication is not None and not publication.store.ref.closed:
            publication.event('close')
            if publication.state.debt:
                raise RuntimeError('Baked Image publication still requires repair')

        def retired():
            if workflow is not None and not workflow.store.ref.closed:
                if workflow.state.phase == 'retiring':
                    workflow.event('retired', accepted=True)
                if workflow.state.phase != 'closed':
                    raise RuntimeError('Baked Image workflow did not retire')
                workflow.close()
            if publication is not None and not publication.store.ref.closed:
                publication.close()
            state['retired'] = True
            _release_reservation(state)
        return retired

    def _drain_committed(token, state):
        nonlocal _last_error
        try:
            _discard_repair(state)
            if 'retirement' not in state:
                state['retirement'] = _retirement(state)
            if not state.get('retired') and not state.get('release_registered'):
                capture.release_bake(token, on_retired=state['retirement'])
                state['release_registered'] = True
            if not state.get('retired'):
                return False
            _release_state(state)
            _states.pop(token, None)
            return True
        except Exception as error:
            state['cleanup_error'] = _last_error = 'Baked Image cleanup requires retry: ' + str(error)
            return False

    def drain_committed():
        """Retry cleanup without changing an already committed native Image."""
        pending = False
        for token, state in tuple(_states.items()):
            if state.get('committed') and not _drain_committed(token, state):
                pending = True
        for state in tuple(_fenced_states.values()):
            if not _finish_fenced(state):pending = True
        return not pending

    def _wake_cleanup():
        nonlocal _last_error
        try:
            if scheduler.lifecycle_admits('timer'):
                scheduler.wake(poll, .05)
        except Exception as error:
            _last_error = 'Baked Image cleanup requires retry: ' + str(error)

    def _discard_image(image):
        try:
            if image.users - int(image.use_fake_user) == 0:
                image.use_fake_user = False
                bpy.data.images.remove(image)
        except ReferenceError:
            pass  # A successful attachment rollback already removed it.

    def _finish_fenced(state):
        if not state.get('physical_retired') or state.get('replacement_wait'):
            return False
        publication = state.get('publication')
        if publication is not None and not publication.store.ref.closed:
            if publication.state.debt:
                publication.require('relinquished', 'release', validated=True)
            publication.event('close')
            publication.close()
        workflow = state.get('workflow')
        if workflow is not None and not workflow.store.ref.closed:
            workflow.event('close')
            if workflow.state.phase == 'retiring':
                workflow.event('retired', accepted=True)
            workflow.close()
        _release_reservation(state)
        _release_state(state)
        _fenced_states.pop(state['token'], None)
        return True

    def _capture_retired(state):
        if state.get('fenced'):
            state['physical_retired'] = True
            _finish_fenced(state)
        else:
            _retirement(state)()

    def fence_runtime(native=False):
        """Retire runtime ownership without restoring native bytes or graphs."""
        if _queue is not None and not _queue.store.ref.closed:
            _queue.event('close')
        for token, state in tuple(_states.items()):
            if state.get('committed'):
                _drain_committed(token, state)
                continue
            _states.pop(token, None)
            _remove_timer(token)
            state['fenced'] = True
            state['replacement_wait'] = bool(native)
            _fenced_states[token] = state
            state['workflow'].event('close')
            state['publication'].event('close')
            def retired(state=state):
                state['physical_retired'] = True
                _finish_fenced(state)
            # Replace the ordinary rollback/close callback before cancellation
            # can dispose an already completed capture's temporary files.
            try:
                capture.on_bake_retired(token, retired)
            except KeyError:
                # The original callback routes through _capture_retired even
                # if another cleanup owner already moved this physical lease.
                if state.get('retired'):
                    state['physical_retired'] = True
            capture.cancel_bake(token)
            capture.release_bake(token, on_retired=retired)
        if not native:
            for state in tuple(_fenced_states.values()):
                state['replacement_wait'] = False
                _finish_fenced(state)
        scheduler.unwake(poll)

    def physical_retired():
        drain_committed()
        for state in tuple(_fenced_states.values()):
            _finish_fenced(state)
        # Committed publication cleanup may still own a helper receipt. Its
        # registration failure must not look like completed runtime recovery.
        return not _states and all(state.get('physical_retired', False)
                                   for state in _fenced_states.values())

    def _recovery_event(record, kind, **values):
        store = record['store']
        application.app.require_store(store)
        return store.machine.event_type(kind=kind, generation=record['observation'], **values)

    def _begin_observation(record):
        record['observation'] = next(_recovery_observations)
        record['store'].dispatch(_recovery_event(record, 'begin'))
        record.pop('error', None)

    def begin_replacement():
        """Record scalar publication IDs, then fence work before native Undo/load."""
        nonlocal _replacement_pending, _recovery_completed, _recovery_total
        _replacement_pending = True
        _recovery_completed = 0
        _recovery_outcomes.clear()
        for image in bpy.data.images:
            key = int(image.session_uid)
            if not (image.get(TAG) or image.get('_pg_payload_record')) and key not in _replacement_images:
                continue
            commit = _commit(image)
            record = _replacement_images.get(key)
            if record is None:
                record = dict(store=application.app.store(machine=application.machines['bake_image']),
                              before=commit, managed=commit is not None, observation=0)
                _replacement_images[key] = record
            elif record['before'] != commit:
                record['managed'] = record['managed'] or commit is not None
                record['store'].dispatch(_recovery_event(record, 'observe', changed=True,
                    managed=record['managed'], dirty=bool(image.is_dirty)))
            _begin_observation(record)
            record['before'] = commit
        _recovery_total = len(_replacement_images)
        fence_runtime(native=True)

    def retire_replaced_state():
        """A real post-replacement receipt supersedes old native ownership."""
        nonlocal _replacement_pending
        if not _replacement_pending:
            return
        _replacement_pending = False
        for state in tuple(_fenced_states.values()):
            if not state.get('replacement_wait'):
                continue
            publication = state['publication']
            if not publication.store.ref.closed and publication.state.debt:
                publication.require('superseded', 'release',
                    generation=publication.identity['generation'] + 1, validated=True)
            state['replacement_wait'] = False
            _finish_fenced(state)

    def runtime_reconcile(max_items=1):
        """Settle bounded native-cache operations; never inspect Image content."""
        nonlocal _recovery_completed
        present = {int(image.session_uid): image for image in bpy.data.images}
        processed = 0
        for key, record in tuple(_replacement_images.items()):
            if processed >= max_items:
                break
            processed += 1
            store = record['store']
            if store.state.phase != 'observing':
                _begin_observation(record)
            observation = record['observation']
            image = present.get(key)
            try:
                if image is None:
                    effects = store.dispatch(_recovery_event(record, 'observe', present=False))
                else:
                    commit = _commit(image)
                    managed = (record['managed'] or commit is not None) and image.packed_file is not None
                    effects = store.dispatch(_recovery_event(record, 'observe',
                        changed=record['before'] != commit, managed=managed, dirty=bool(image.is_dirty)))
                    if effects.invalidate:
                        # Native operation return is the receipt. It says only
                        # that the stale decoded cache was discarded, not that
                        # any particular pixel data is valid or was read back.
                        receipt = store.event(_recovery_event(record, 'invalidated', matched=True))
                        with guard.publication():
                            if (_replacement_images.get(key) is not record or
                                    record['observation'] != observation or
                                    int(image.session_uid) != key or _commit(image) != commit):
                                raise ReferenceError('Baked Image recovery observation was superseded')
                            if image.is_dirty or image.packed_file is None:
                                effects = store.dispatch(_recovery_event(record, 'observe',
                                    managed=image.packed_file is not None, dirty=bool(image.is_dirty)))
                            else:
                                image.buffers_free()
                                _checkpoint('recovery_invalidated')
                                if (_replacement_images.get(key) is not record or
                                        record['observation'] != observation or
                                        int(image.session_uid) != key or _commit(image) != commit):
                                    raise ReferenceError('Baked Image recovery observation was superseded')
                                if image.is_dirty:
                                    effects = store.dispatch(_recovery_event(record, 'observe',
                                        managed=managed, dirty=True))
                                else:
                                    effects = store.dispatch(receipt)
                if not effects.settled:
                    raise RuntimeError('Baked Image reconciliation receipt was rejected')
                _recovery_outcomes[key] = store.state.phase
                _replacement_images.pop(key, None)
                _recovery_completed += 1
                store.close()
            except guard.PublicationDeferred:
                raise
            except Exception as error:
                if (_replacement_images.get(key) is record and record['observation'] == observation):
                    store.dispatch(_recovery_event(record, 'failed'))
                    record['error'] = str(error)
                return dict(done=False, completed=_recovery_completed, total=_recovery_total, error=str(error))
        return dict(done=not _replacement_images, completed=_recovery_completed,
                    total=_recovery_total, error='')

    def recovery_pending():
        return bool(_replacement_images)

    def _release_state(state):
        repair = state.pop('repair', None)
        if repair is not None:
            ownership.release(repair['image'])
        for key in ('target', 'binding', 'kernel'):
            ownership.release(state.pop(key, None))

    def _resolve_target(state):
        node = ownership.resolve(state['target'])
        if node is None or not _valid(node):
            raise ReferenceError('The bake target was removed or replaced')
        kernel = ownership.resolve(state['kernel'])
        image = ownership.resolve(state['binding'])
        if node.node_tree != kernel or baked_image(node) != image:
            raise ReferenceError('The bake target binding changed')
        if state['binding'] is not None and image is None:
            raise ReferenceError('The bound bake Image was removed')
        return node, image

    def image_from_artifact(artifact):
        """Load and pack one helper artifact under the publication lease."""
        path = Path(artifact['path'])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('Bake artifact is unavailable')
        if _sha256(path) != str(artifact.get('sha256', '')):
                    raise ValueError('Bake artifact changed before Image creation')
        image = None
        try:
            guard.require_available()
            with guard.publication():
                image = bpy.data.images.load(str(path), check_existing=False)
                if image is None or image.size[0] < 1 or image.size[1] < 1:
                    raise ValueError('Bake produced no pixels')
                if tuple(image.size) != (int(artifact['width']), int(artifact['height'])):
                    raise ValueError('Bake artifact dimensions changed')
                if not image.is_float:
                    raise ValueError('Bake requires a float Image')
                image.name = _unique_name(
                    'Perceptual Grade Bake · ' + str(artifact.get('node', 'Color')))
                image.colorspace_settings.name = 'Non-Color'
                image.alpha_mode = 'CHANNEL_PACKED'
                if hasattr(image, 'use_half_precision'):
                    image.use_half_precision = False
                _write_operation(image, uuid.uuid4().hex, 'WRITING')
                _metadata(image, artifact)
                _pack(image)
                # The user assigns the Image later. Keep an unassigned bake in
                # the saved project, including its packed full-precision EXR.
                image.use_fake_user = True
            return image
        except Exception:
            if image is not None and image.users - int(image.use_fake_user) == 0:
                image.use_fake_user = False
                bpy.data.images.remove(image)
            raise

    def finish(token):
        """Publish synchronously inside the caller's undoable operation.

        App timers only stage immutable artifact descriptions. The UI invokes
        this from its final modal TIMER callback immediately before FINISHED,
        so Blender snapshots the packed Image and metadata together.
        """
        nonlocal _last_error
        if _cancel_for_playback() and (token not in _states or token in _playback_pending):
            return None
        state = _states.get(token)
        if state is not None and not _inputs_current(state, force=True):
            _interrupt_inputs(token, state)
            return None
        artifact = status(token)
        if artifact is None:
            return None
        candidate = None
        image = None
        node = None
        attached = False
        success = False
        state = _states[token]
        workflow, publication = state['workflow'], state['publication']
        try:
            node, image = _resolve_target(state)
            guard.require_available()
            candidate = image_from_artifact(artifact)
            with guard.publication():
                if not _inputs_current(state, native_lease=True, force=True):
                    state['interruption'] = 'bake.inputs_changed'
                    raise RuntimeError('Bake inputs changed before publication')
                operation = _operation(candidate)['id']
                node, image = _resolve_target(state)
                if image is None:
                    kernel = node.node_tree
                    state['repair'] = dict(kind='attach', image=ownership.retain(candidate, 'bake.repair.image'))
                    workflow.require('publish', 'publish', candidate=state['candidate'], accepted=True)
                    _queue_require(state, 'publish', 'publish', validated=True)
                    publication.require('begin', 'reserve')
                    publication.require('reserved', 'write', accepted=True)
                    publication.require('written', 'verify')
                    publication.require('verified', 'publish', validated=True, readable=True)
                    state['mutation_started'] = True
                    set_baked_image(node, candidate)
                    attached = True
                    image = candidate
                    _checkpoint('attached')
                else:
                    if not is_baked_image(image):
                        raise ValueError('Bake destination must be a packed Perceptual Grade Image')
                    if image.packed_file is None or image.is_dirty:
                        raise ValueError('Pack or save pending baked Image edits before rebaking')
                    metadata = {key: value.to_list() if hasattr(value, 'to_list') else value
                                for key, value in image.items() if key.startswith('_pg_bake_')}
                    state['repair'] = dict(kind='overwrite', bytes=bytes(image.packed_file.data),
                        metadata=metadata,
                        image=ownership.retain(image, 'bake.repair.image'))
                    workflow.require('publish', 'publish', candidate=state['candidate'], accepted=True)
                    _queue_require(state, 'publish', 'publish', validated=True)
                    publication.require('begin', 'reserve')
                    publication.require('reserved', 'write', accepted=True)
                    state['mutation_started'] = True
                    _write_operation(image, operation, 'WRITING')
                    _write_packed(image, bytes(candidate.packed_file.data))
                    _checkpoint('packed')
                    publication.require('written', 'verify')
                    _checkpoint('transferred')
                    publication.require('verified', 'publish', validated=True, readable=True)
                    _metadata(image, artifact)
                    _checkpoint('metadata')
                    image.update()
                    image.update_tag()
                    node.id_data.update_tag()
                _write_operation(image, operation, 'COMPLETE')
            if not workflow.event('published', candidate=state['candidate'], accepted=True).accepted:
                raise RuntimeError('Baked Image workflow receipt was rejected')
            if not publication.event('published', validated=True).accepted:
                raise RuntimeError('Baked Image publication receipt was rejected')
            # This is the commit point. Later cleanup only releases ownership;
            # it can neither change these pixels nor cancel the undoable result.
            state['committed'] = True
            _last_error = ''
            success = True
            _errors.pop(state['target_key'], None)
            if not _drain_committed(token, state):
                _wake_cleanup()
        except Exception as error:
            if isinstance(error, guard.PublicationDeferred) and state.get('repair') is None:
                raise  # No mutation occurred; keep the artifact for another turn.
            _error(state, error)
            workflow.event('failed')
            try:
                _restore(state)
            except Exception as repair_error:
                _error(state, 'Baked Image rollback requires repair: ' + str(repair_error))
                _wake_cleanup()
                raise RuntimeError(_last_error) from error
            cancel(token)
            if state.get('interruption') == 'bake.inputs_changed':
                return None
            raise
        finally:
            retained = state.get('repair', {}).get('kind') == 'attach'
            if candidate is not None and not (success and attached) and not retained:
                _discard_image(candidate)
        try:
            redraw()
        except (ReferenceError, RuntimeError) as error:
            _last_error = 'Baked Image redraw requires retry: ' + str(error)
        return image

    def poll():
        """Observe helper readiness without publishing Blender datablocks."""
        nonlocal _last_error
        _cancel_for_playback()
        _cancel_for_inputs()
        if guard.available():
            sync_coordinates()
        pending = False
        for token, state in tuple(_states.items()):
            if state.get('committed'):
                if not _drain_committed(token, state):
                    pending = True
                continue
            if state.get('repair') is not None:
                try:
                    _restore(state)
                    cancel(token)
                except Exception:
                    pending = True
                continue
            if state.get('artifact') is not None or state.get('error'):
                continue
            try:
                artifact = capture.bake_status(token)
            except Exception as error:
                changed = getattr(capture, 'BakeInputsChanged', ())
                if changed and isinstance(error, changed):
                    _interrupt_inputs(token, state)
                    continue
                _error(state, error)
                redraw()
                continue
            if artifact is None:
                pending = True
                continue
            _queue_require(state, 'complete', 'stage', validated=True)
            effect = state['workflow'].dispatch(state['candidate_receipt'])
            if not effect.accepted or effect.action != 'stage':
                state['error'] = _last_error = 'Baked Image candidate was superseded'
                continue
            state['artifact'] = artifact
        _dispatch()
        # Ready artifacts still need input monitoring while a native reader
        # prevents their modal operator from committing them.
        return .05 if pending or _playback_pending or _states else None

    def start(node, context=None):
        nonlocal _last_error, _last_error_node
        _cancel_for_playback(context)
        guard.reconcile()
        if playback_active(context):
            return None
        observation = eligibility(node, context)
        if observation['state'] != 'ready':
            _blocked(node, observation)
            return None
        if _is_shader(node):
            routes = application.component('scopes.bake_routes')
            try:
                # Full revalidation precedes every reservation. A cached UI
                # observation never authorizes a different source snapshot.
                routes.describe(node, context)
            except routes.RouteBlocked as error:
                _blocked(node, dict(state='blocked', code=error.code))
                return None
        image = baked_image(node)
        if image is not None and not image.is_float:
            _blocked(node, dict(state='blocked', code='destination'))
            return None
        _last_error_node = _node_key(node)
        pool = _pool()
        slot = next((index for index, entry in enumerate(pool.state.entries) if not entry.request), None)
        if slot is None:
            _last_error = 'Bake queue is full (8 requests)'
            _errors[_last_error_node] = _last_error
            raise RuntimeError(_last_error)
        image = baked_image(node)
        try:
            if image is not None:
                if not is_baked_image(image):
                    raise ValueError('Bake destination must be a packed Perceptual Grade Image')
                if image.is_dirty:
                    raise ValueError('Pack or save pending baked Image edits before rebaking')
        except Exception as error:
            _last_error = str(error)
            _errors[_last_error_node] = _last_error
            raise
        state = {'artifact': None, 'error': '',
                 'input_revision': nodes.observe_intent(node),
                 'target': ownership.retain(node, 'bake.target'),
                 'kernel': ownership.retain(node.node_tree, 'bake.kernel'),
                 'binding': ownership.retain(image, 'bake.image'), 'target_key': _node_key(node),
                 'queue': pool}
        operation = next(_operations)
        state['reservation'] = dict(slot=slot, request=operation, target=int(node.as_pointer()),
            destination=int(image.session_uid) * 2 if image is not None else int(node.as_pointer()) * 2 + 1)
        _queue_require(state, 'admit', 'freeze')
        identity = {'generation': 1, 'revision': operation, 'target': int(node.as_pointer()), 'mode': 'manual'}
        workflow = state['workflow'] = Protocol(application, 'bake', identity=identity)
        state['candidate'] = operation
        state['publication'] = Protocol(application, 'publication', identity={
            'generation': 1, 'operation': operation, 'resource': operation, 'target': identity['target']})
        token = None
        try:
            workflow.require('request', 'admit')
            workflow.require('admitted', 'compute', accepted=True)
            state['candidate_receipt'] = workflow.receipt('candidate', candidate=operation, accepted=True)
            token = capture.start_bake(node, context=context, defer_submit=True)
            state['token'] = token
            _queue_require(state, 'frozen', 'queue', validated=True)
            _states[token] = state
            # Register while the captured token is owned. A later global clear
            # can then retire it even if foreground release admission fails.
            capture.on_bake_retired(token, lambda: _capture_retired(state))
            if not _inputs_current(state):
                _interrupt_inputs(token, state)
                return None
        except Exception as error:
            changed = getattr(capture, 'BakeInputsChanged', ())
            if changed and isinstance(error, changed):
                state['interruption'] = 'bake.inputs_changed'
            if not isinstance(error, guard.PublicationDeferred):
                _error(state, error)
            _queue_event(state, 'cancel')
            if token is None:
                _retirement(state)()
                _release_state(state)
            else:
                cancel(token)
            if isinstance(error, guard.PublicationDeferred):
                return None
            if state.get('interruption') == 'bake.inputs_changed':
                return None
            raise
        _last_error = ''
        _errors.pop(state['target_key'], None)
        _interruptions.discard(state['target_key'])
        application.component('integration.outcomes').clear(node, 'bake')
        _dispatch()
        try:
            scheduler.wake(poll, .05)
        except Exception:
            cancel(token)
            raise
        return token

    def cancel(token=None):
        for key in tuple(_states if token is None else (token,)):
            _remove_timer(key)
            state = _states.get(key)
            if state is not None:
                if state.get('committed'):
                    _drain_committed(key, state)
                    continue
                _restore(state)
                _queue_event(state, 'cancel')
                _states.pop(key, None)
                try:
                    capture.cancel_bake(key)
                finally:
                    # Release removes admission, not a live helper's files.
                    try:
                        capture.release_bake(key, on_retired=_retirement(state))
                    finally:
                        _release_state(state)

    def status(token):
        state = _states.get(token)
        if state is None:
            raise KeyError('Unknown bake request')
        if state.get('error'):
            raise ValueError(state['error'])
        artifact = state.get('artifact')
        return dict(artifact) if artifact is not None else None

    def _remove_timer(token):
        timer = _timers.pop(token, None)
        if timer is not None:
            try:
                bpy.context.window_manager.event_timer_remove(timer)
            except (ReferenceError, RuntimeError):
                pass

    def clear():
        """Clear runtime owners while preserving Blender's authoritative result."""
        nonlocal _last_error, _last_error_node
        fence_runtime(native=_replacement_pending)
        for token in tuple(_timers):
            _remove_timer(token)
        _last_error = ''
        _last_error_node = None
        _errors.clear()
        _interruptions.clear()
        _playback_pending.clear()

    def busy(node=None):
        if node is None:
            return bool(_states)
        return bool(_queue is not None and not _queue.store.ref.closed and not _queue.state.closing
                    and _queue.store.query('target_busy', target=int(node.as_pointer())))

    def ready(node, context=None):
        """Observe the same admission boundary for drawing, polling and clicks."""
        return eligibility(node, context)['state'] == 'ready'

    def picker_ready(node, context=None):
        reader = getattr(node, 'bl_idname', '') in {'PGShaderBake', 'PGCompositorBake'}
        kernel = getattr(node, 'node_tree', None)
        return bool(not playback_active(context) and guard.available()
            and node is not None and _editable_tree(node.id_data) and (_valid(node) or reader)
            and not busy(node) and kernel is not None and grades.local(kernel)
            and (reader or kernel.users - int(kernel.use_fake_user) <= 1))

    def eligibility(node, context=None):
        """Read-only admission shared by the button, hover, poll and action."""
        if not guard.available():
            return dict(state='waiting', code='waiting', route=None)
        if playback_active(context):
            return dict(state='waiting', code='playback', route=None)
        if not _valid(node):
            return dict(state='blocked', code='target', route=None)
        if busy(node):
            return dict(state='waiting', code='busy', route=None)
        kernel = getattr(node, 'node_tree', None)
        if (kernel is None or not output_current(kernel) or not grades.local(kernel)
                or kernel.users - int(kernel.use_fake_user) > 1):
            return dict(state='blocked', code='resources', route=None)
        image = baked_image(node)
        if image is not None and not _image_allowed(node, image):
            return dict(state='blocked', code='destination', route=None)
        if _image_source(kernel).image != image:
            return dict(state='blocked', code='resources', route=None)
        if image is not None and image.is_dirty:
            return dict(state='blocked', code='dirty', route=None)
        if (_queue is not None and not _queue.store.ref.closed
                and not any(not entry.request for entry in _queue.state.entries)):
            return dict(state='waiting', code='queue', route=None)
        if _is_shader(node):
            observation = application.component('scopes.bake_routes').observe(node, context)
            if observation['state'] == 'ready':
                return _preflight_observation(node, context, observation)
            return observation
        return dict(state='ready', code='', route=None)

    def _blocked(node, observation, control='bake'):
        if node is not None and observation['code'] and observation.get('state') != 'waiting':
            application.component('integration.outcomes').set(
                node, control, 'bake.blocked.' + observation['code'])

    def draw_image_selector(layout, node, context=None):
        row = layout.row(align=True)
        row.enabled = picker_ready(node, context)
        row.context_pointer_set('pg_node', node)
        if baked_image(node) is None:
            row.template_ID(node, 'pg_bake_image', new='perceptual_grade.new_bake_image')
        else:
            row.template_ID(node, 'pg_bake_image', unlink='perceptual_grade.unlink_bake_image')

    def redraw():
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()

    def draw_action(layout, node, context=None):
        if not (_is_compositor(node) or _is_shader(node)):
            return
        layout.context_pointer_set('pg_node' if nodes.is_grade(node) else 'pg_stack_node', node)
        draw_image_selector(layout, node, context)
        observation = eligibility(node, context)
        row = layout.row(align=True)
        row.enabled = observation['state'] == 'ready'
        row.operator('perceptual_grade.bake_compositor', text=tr('bake.label'), icon='IMAGE_DATA')
        if _is_shader(node):
            route = observation['route']
            if route is not None:
                layout.label(text=tr('bake.route.uv' if route['domain'] == 'UV' else 'bake.route.image'),
                             icon='IMAGE_DATA')
                reason_labels = {
                    'linked_coordinates': 'bake.route.linked_coordinates',
                    'surface_dependency': 'bake.route.surface_dependency',
                    'multiple_images': 'bake.route.multiple_images',
                    'existing_uv_binding': 'bake.route.existing_uv_binding',
                }
                for reason in route['reasons']:
                    layout.label(text=tr(reason_labels[reason]))
                if route['domain'] == 'IMAGE':
                    layout.label(text=tr('bake.route.constant' if route['source'] == 'constant'
                                         else 'bake.route.image_source'))
                if route['show_dimensions']:
                    row = layout.row(align=True)
                    if (baked_image(node) is not None
                            and baked_image(node).get('_pg_bake_kind') != 'EMPTY'):
                        row.label(text=f"W {route['width']}  H {route['height']}")
                    else:
                        row.enabled = picker_ready(node, context)
                        row.prop(node, 'pg_bake_width', text='W')
                        row.prop(node, 'pg_bake_height', text='H')
        if baked_image(node) is None:
            layout.label(text=tr('bake.empty'), icon='IMAGE_DATA')
        active = _active_state(node)
        if active is not None:
            phase = active['queue'].state.entries[active['reservation']['slot']].phase
            label = 'bake.queued' if phase in ('freezing', 'queued') else 'bake.publish_wait' if phase == 'ready' else 'bake.status'
            layout.label(text=tr(label), icon='TIME')
        if observation['state'] == 'blocked':
            layout.label(text=tr('bake.blocked.' + observation['code']), icon='INFO')
        application.component('integration.outcomes').draw(layout, node)

    def _kernel(node):
        shader = node.bl_idname == 'PGShaderBake'
        tree = bpy.data.node_groups.new('Perceptual Grade Bake · Kernel',
                                        'ShaderNodeTree' if shader else 'CompositorNodeTree')
        tree[KERNEL] = SCHEMA
        if shader:
            vector = tree.interface.new_socket(name='Vector', in_out='INPUT', socket_type='NodeSocketVector')
            vector.default_value = (0., 0., 0.)
            vector.description = native('bake.vector.tooltip')
            color = tree.interface.new_socket(name='Color', in_out='OUTPUT', socket_type='NodeSocketColor')
            alpha = tree.interface.new_socket(name='Alpha', in_out='OUTPUT', socket_type='NodeSocketFloat')
        else:
            output_socket = tree.interface.new_socket(name='Color', in_out='OUTPUT',
                                                      socket_type='NodeSocketColor')
        source = tree.nodes.new('ShaderNodeTexImage' if shader else 'CompositorNodeImage')
        source.name = 'Baked Image'
        source[SOURCE] = 1
        output = tree.nodes.new('NodeGroupOutput')
        output.is_active_output = True
        if shader:
            group_input = tree.nodes.new('NodeGroupInput')
            texcoord = tree.nodes.new('ShaderNodeTexCoord'); texcoord['_pg_bake_uv_fallback'] = 1
            tree.links.new(texcoord.outputs['UV'], source.inputs['Vector'])
            tree.links.new(source.outputs['Color'], output.inputs[color.identifier])
            tree.links.new(source.outputs['Alpha'], output.inputs[alpha.identifier])
        else:
            tree.links.new(source.outputs.get('Image'), output.inputs[output_socket.identifier])
        return tree

    class PGCompositorBake(bpy.types.CompositorNodeCustomGroup):
        bl_idname = 'PGCompositorBake'
        bl_label = native('bake.node.label')
        bl_description = native('bake.node.tooltip')

        def init(self, context):
            # Blender 5.2 increments ID users while a node is still inside init;
            # the explicit add operator assigns storage only after nodes.new().
            self['_pg_bake_pending_init'] = True
            self.width = 220

        def copy(self, original):
            if self.id_data.is_evaluated:
                return
            self['_pg_bake_pending_copy'] = True
            if (original.node_tree is not None and any(
                    item.as_pointer() == original.as_pointer() for item in self.id_data.nodes)):
                self.node_tree = original.node_tree.copy()
                del self['_pg_bake_pending_copy']

        def free(self):
            if self.id_data.is_evaluated:
                return
            pointer = self.as_pointer()
            _freeing.add(pointer)
            try:
                # Blender does not decrement extension PointerProperty users
                # during individual node removal. Never mutate a shared kernel.
                with binding_suppressed(self):
                    self.pg_bake_image = None
            finally:
                _freeing.discard(pointer)

        def draw_buttons(self, context, layout):
            draw_image_selector(layout, self, context)
            application.component('integration.outcomes').draw(layout, self)

    class PGShaderBake(bpy.types.ShaderNodeCustomGroup):
        bl_idname = 'PGShaderBake'
        bl_label = native('bake.node.label')
        bl_description = native('bake.node.tooltip')

        def init(self, context):
            self['_pg_bake_pending_init'] = True
            self.width = 220

        def copy(self, original):
            if self.id_data.is_evaluated:
                return
            self['_pg_bake_pending_copy'] = True
            if (original.node_tree is not None and any(
                    item.as_pointer() == original.as_pointer() for item in self.id_data.nodes)):
                self.node_tree = original.node_tree.copy()
                del self['_pg_bake_pending_copy']

        def free(self):
            if self.id_data.is_evaluated:
                return
            pointer = self.as_pointer(); _freeing.add(pointer)
            try:
                with binding_suppressed(self):
                    self.pg_bake_image = None
            finally:
                _freeing.discard(pointer)

        def draw_buttons(self, context, layout):
            draw_image_selector(layout, self, context)
            application.component('integration.outcomes').draw(layout, self)

    class PG_OT_bake_compositor(bpy.types.Operator):
        bl_idname = 'perceptual_grade.bake_compositor'
        bl_label = native('bake.label')
        bl_description = native('bake.tooltip')
        bl_options = {'REGISTER', 'UNDO'}
        token: StringProperty(options={'SKIP_SAVE'})

        def _begin(self, context):
            if playback_active(context):
                _cancel_for_playback(context)
                return {'CANCELLED'}
            try:
                guard.reconcile()
                observation = eligibility(_target(context), context)
                if observation['state'] != 'ready':
                    _blocked(_target(context), observation)
                    return {'CANCELLED'}
                if context.window is None:
                    _blocked(_target(context), dict(code='window'))
                    return {'CANCELLED'}
                token = start(_target(context), context=context)
                if token is None:
                    return {'CANCELLED'}
                self.token = token
                _timers[self.token] = context.window_manager.event_timer_add(.1, window=context.window)
                context.window_manager.modal_handler_add(self)
            except Exception as error:
                cancel(getattr(self, 'token', ''))
                routes = application.component('scopes.bake_routes')
                outcomes = application.component('integration.outcomes')
                if isinstance(error, routes.RouteBlocked):
                    outcomes.set(_target(context), 'bake', 'bake.blocked.' + error.code)
                else:
                    outcomes.failure(_target(context), 'bake', error,
                                     recovery='perceptual_grade.bake_compositor')
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}

        def invoke(self, context, event):
            return self._begin(context)

        @classmethod
        def poll(cls, context):
            observation = eligibility(_target(context), context)
            if observation['code']:
                cls.poll_message_set(i18n.tip('bake.blocked.' + observation['code']))
            return observation['state'] == 'ready'

        @classmethod
        def description(cls, context, properties):
            observation = eligibility(_target(context), context)
            return i18n.tip('bake.blocked.' + observation['code'] if observation['code'] else 'bake.tooltip')

        def execute(self, context):
            return self._begin(context)

        def modal(self, context, event):
            _cancel_for_playback(context)
            if self.token not in _states:
                _remove_timer(self.token)
                return {'CANCELLED'}
            if event.type == 'ESC':
                cancel(self.token)
                return {'CANCELLED'}
            if event.type != 'TIMER':
                return {'PASS_THROUGH'}
            poll()
            if self.token not in _states:
                return {'CANCELLED'}
            try:
                image = finish(self.token)
            except guard.PublicationDeferred:
                return {'PASS_THROUGH'}
            except Exception as error:
                cancel(self.token)
                application.component('integration.outcomes').failure(
                    _target(context), 'bake', error, recovery='perceptual_grade.bake_compositor')
                return {'CANCELLED'}
            if image is not None:
                application.component('integration.outcomes').clear(_target(context), 'bake')
                _remove_timer(self.token)
                return {'FINISHED'}
            # Blender does not expose the originating Timer on Event. Let
            # every pending modal bake receive ticks, including an earlier
            # request whose destination the newer request is waiting for.
            return {'PASS_THROUGH'}

        def cancel(self, context):
            cancel(self.token)

    class PG_OT_new_bake_image(bpy.types.Operator):
        bl_idname = 'perceptual_grade.new_bake_image'
        bl_label = native('bake.new')
        bl_description = native('bake.new.tooltip')
        bl_translation_context = i18n.CONTEXT
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            node = _target(context)
            return picker_ready(node, context) and baked_image(node) is None

        def execute(self, context):
            node = _target(context)
            image = new_image(node)
            if image is None:
                return {'CANCELLED'}
            application.component('integration.outcomes').clear(node, 'image')
            return {'FINISHED'}

        @classmethod
        def description(cls, context, properties):
            return i18n.tip('bake.new.tooltip')

    class PG_OT_unlink_bake_image(bpy.types.Operator):
        bl_idname = 'perceptual_grade.unlink_bake_image'
        bl_label = native('bake.unlink')
        bl_description = native('bake.unlink.tooltip')
        bl_translation_context = i18n.CONTEXT
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            node = _target(context)
            return picker_ready(node, context) and baked_image(node) is not None

        def execute(self, context):
            guard.reconcile()
            node = _target(context)
            if not picker_ready(node, context):
                return {'CANCELLED'}
            set_baked_image(node, None)
            application.component('integration.outcomes').clear(node, 'image')
            return {'FINISHED'}

        @classmethod
        def description(cls, context, properties):
            return i18n.tip('bake.unlink.tooltip')

    class PG_OT_check_bake_inputs(bpy.types.Operator):
        bl_idname = 'perceptual_grade.check_bake_inputs'
        bl_label = native('bake.check')
        bl_translation_context = i18n.CONTEXT
        bl_options = {'INTERNAL'}

        @classmethod
        def poll(cls, context):
            return _valid(_target(context))

        def execute(self, context):
            node = _target(context)
            target = _node_key(node)
            for key in tuple(_preflights):
                if key[0] == target:
                    _preflights.pop(key, None)
            application.component('integration.outcomes').clear(node, 'bake')
            eligibility(node, context)
            redraw()
            return {'FINISHED'}

    def create_node(tree):
        if not _editable_tree(tree):
            raise ValueError('Bake nodes require a local Shader or Compositor tree')
        node_type = PGShaderBake.bl_idname if tree.bl_idname == 'ShaderNodeTree' else PGCompositorBake.bl_idname
        node = tree.nodes.new(node_type)
        try:
            node.node_tree = _kernel(node)
            del node['_pg_bake_pending_init']
        except Exception:
            tree.nodes.remove(node)
            raise
        return node

    class PG_OT_add_bake(bpy.types.Operator):
        bl_idname = 'perceptual_grade.add_bake'
        bl_label = native('bake.add.label')
        bl_description = native('bake.add.tooltip')
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            space = getattr(context, 'space_data', None)
            tree = getattr(space, 'edit_tree', None) if space else None
            return _editable_tree(tree)

        def execute(self, context):
            tree = context.space_data.edit_tree
            active = tree.nodes.active
            node = create_node(tree)
            if active is not None and active != node:
                node.location = (active.location.x + active.width + 40, active.location.y)
            tree.nodes.active = node
            for item in tree.nodes:
                item.select = item == node
            return {'FINISHED'}

    def menu(self, context):
        self.layout.separator()
        self.layout.operator(PG_OT_add_bake.bl_idname, icon='IMAGE_DATA')

    def register():
        nonlocal _registered, _menu_registered
        bpy.types.Node.pg_bake_image = application.component('integration.outcomes').property(
            image_property(), control='image')
        for cls in (PGCompositorBake, PGShaderBake, PG_OT_bake_compositor, PG_OT_add_bake,
                    PG_OT_new_bake_image, PG_OT_unlink_bake_image, PG_OT_check_bake_inputs):
            application.register_class(cls)
            _classes.append(cls)
        bpy.types.NODE_MT_category_compositor_color.append(menu)
        bpy.types.NODE_MT_category_shader_color.append(menu)
        bpy.types.Node.pg_bake_width = IntProperty(name='Bake Width', min=1, max=16384,
            get=_dimension_get('_pg_bake_width'), set=_dimension_set('_pg_bake_width'), options={'SKIP_SAVE'})
        bpy.types.Node.pg_bake_height = IntProperty(name='Bake Height', min=1, max=16384,
            get=_dimension_get('_pg_bake_height'), set=_dimension_set('_pg_bake_height'), options={'SKIP_SAVE'})
        application.add_handler('animation_playback_pre', playback_started)
        application.add_handler('depsgraph_update_post', preflight_invalidated)
        scheduler.wake(preflight_tick, .1, persistent=True)
        _menu_registered = True
        _registered = True

    def unregister():
        nonlocal _registered, _menu_registered, _last_error
        clear()
        scheduler.unwake(preflight_tick)
        _preflights.clear()
        application.remove_handler('animation_playback_pre', playback_started)
        application.remove_handler('depsgraph_update_post', preflight_invalidated)
        if _menu_registered:
            try:
                bpy.types.NODE_MT_category_compositor_color.remove(menu)
            except (ValueError, RuntimeError):
                pass
            try:
                bpy.types.NODE_MT_category_shader_color.remove(menu)
            except (ValueError, RuntimeError):
                pass
            _menu_registered = False
        for cls in reversed(_classes):
            application.unregister_class(cls)
        _classes.clear()
        if hasattr(bpy.types.Node,'pg_bake_image'):
            del bpy.types.Node.pg_bake_image
        for name in ('pg_bake_width', 'pg_bake_height'):
            if hasattr(bpy.types.Node, name):
                delattr(bpy.types.Node, name)
        _registered = False

    def _set_pack_image(value):
        nonlocal _pack
        _pack = value

    def _set_playback_active(value):
        nonlocal playback_active
        playback_active = value

    def _set_replace_packed(value):
        nonlocal _write_packed
        _write_packed = value

    def _set_checkpoint(value):
        nonlocal _checkpoint
        _checkpoint = value

    return Component({
        '__name__': __name__, 'register': register, 'unregister': unregister,
        'menu': menu,
        'playback_active': playback_active, 'playback_started': playback_started,
        'sync_coordinates': sync_coordinates,
        'start': start, 'poll': poll, 'cancel': cancel, 'status': status,
        'finish': finish, 'clear': clear, 'create_node': create_node,
        'drain_committed': drain_committed,
        'baked_image': baked_image, 'set_baked_image': set_baked_image,
        'is_baked_image': is_baked_image, 'image_allowed': _image_allowed,
        'image_property': image_property, 'image_changed': image_changed,
        'preflight_tick': preflight_tick, 'preflight_invalidated': preflight_invalidated,
        'binding_suppressed': binding_suppressed, 'materialize_binding': materialize_binding,
        'repair_binding': repair_binding, 'verify_binding': verify_binding,
        'repair_image_creation': repair_image_creation, 'verify_image_creation': verify_image_creation,
        'new_image': new_image, 'draw_image_selector': draw_image_selector,
        'create_output': create_output, 'output_current': output_current,
        'begin_replacement': begin_replacement,
        'retire_replaced_state': retire_replaced_state,
        'fence_runtime': fence_runtime, 'physical_retired': physical_retired,
        'runtime_reconcile': runtime_reconcile,
        'recovery_pending': recovery_pending,
        'pack_image': _pack,
        'replace_packed': _write_packed,
        'publication_checkpoint': _checkpoint,
        'busy': busy, 'ready': ready, 'picker_ready': picker_ready, 'eligibility': eligibility,
        'draw_action': draw_action, 'image_from_artifact': image_from_artifact,
        'PGCompositorBake': PGCompositorBake, 'PGShaderBake': PGShaderBake,
        'PG_OT_bake_compositor': PG_OT_bake_compositor,
        'PG_OT_add_bake': PG_OT_add_bake,
        'PG_OT_new_bake_image': PG_OT_new_bake_image,
        'PG_OT_unlink_bake_image': PG_OT_unlink_bake_image,
    }, readers={'states': lambda: _states, 'timers': lambda: _timers,
                'recovery_states': lambda: _replacement_images,
                'recovery_outcomes': lambda: _recovery_outcomes,
                'fenced_states': lambda: _fenced_states,
                'queue': lambda: _queue,
                'last_error': lambda: _last_error}, writers={'pack_image': _set_pack_image,
                    'playback_active': _set_playback_active,
                                                            'replace_packed': _set_replace_packed,
                                                            'publication_checkpoint': _set_checkpoint})
