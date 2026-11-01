"""Native payload adapter for fixed LUT banks, with detached runtime authority.

Resource construction belongs to lut.make_kernel/clone_kernel in an undoable
lifecycle operation. This module never changes native ID relationships.
"""

from ..integration.components import Component


def create(_application):
    from itertools import count
    import hashlib
    import json
    import uuid

    import bpy
    import numpy as np

    generation_key = _application.component('ownership').generation_key; retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    from ...perceptual_grade_core.graph import output_policy as policy

    LAYOUT = 2
    SELECTOR = '_pg_output_selector'
    SLOT = '_pg_output_slot'
    LAYOUT_KEY = '_pg_output_layout'
    REVISION = '_pg_payload_revision'
    RECORD = '_pg_payload_record'
    RECORD_VERSION = 1
    _records = {}
    _serial = count(1)
    _session = _application.app.identity


    class ResourceRecord(dict):
        """Keep callers' record interface while state lives only in its Store."""
        def __getitem__(self, key):
            if key == 'state':
                return super().__getitem__('store').state
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == 'state':
                return self['state']
            return super().get(key, default)

        def __setitem__(self, key, value):
            if key == 'state':
                raise TypeError('Output authority state is committed only by its Store')
            return super().__setitem__(key, value)


    def dispatch(current, event, **values):
        store = current['store']
        _application.app.require_store(store)
        event_type = store.machine.event_type
        effects = store.dispatch(event_type(kind=event, **values))
        return store.state, effects.accepted


    def _require_token_owner(current, token):
        """Transient receipts belong to one App, Store, and incarnation."""
        store = current['store']
        _application.app.require_store(store)
        if (type(token.get('store')) is not int or token['store'] != store.identity
                or type(token.get('session')) is not str or token['session'] != _session
                or type(token.get('incarnation')) is not int
                or token['incarnation'] != store.state.incarnation):
            raise RuntimeError('LUT resource token belongs to another Store or incarnation')
        if ('operation_id' in token and token['operation_id'] != current.get('operation_id')):
            raise RuntimeError('LUT resource token belongs to another producer')
        if 'operation_id' in token:
            state = store.state
            if (state.pending != token.get('index') or state.operation != token.get('operation')
                    or state.pending_source != token.get('source')):
                raise RuntimeError('LUT producer operation is no longer current')


    def is_current(node):
        tree = node.node_tree
        return bool(tree and tree.get(LAYOUT_KEY) == LAYOUT)


    def create_selector(tree):
        selector = tree.nodes.new('ShaderNodeMath')
        selector.operation = 'ADD'
        selector[SELECTOR] = True
        selector.label = 'Output Slot'
        selector.inputs[0].default_value = -1.
        selector.inputs[1].default_value = 0.
        return selector.outputs[0]


    def selector_socket(node):
        if not is_current(node):
            raise ValueError('Invalid output resource layout')
        matches = [item for item in node.node_tree.nodes if item.get(SELECTOR)]
        if (len(matches) != 1 or matches[0].bl_idname != 'ShaderNodeMath'
                or matches[0].operation != 'ADD' or matches[0].inputs[0].is_linked
                or matches[0].inputs[1].is_linked or matches[0].inputs[1].default_value != 0.):
            raise ValueError('Invalid private output selector')
        return matches[0].inputs[0]


    def adopt_native(node, intent):
        """Observe restored ownership without certifying or modifying native data."""
        current = record(node.node_tree, node)
        state = current['state']
        _, accepted = dispatch(current, 'adopt_native', session=state.session,
            incarnation=state.incarnation, slot=selected_index(node),
            intent=json.dumps(intent, sort_keys=True, separators=(',', ':')))
        if not accepted:raise RuntimeError('Native adoption requires retired producer work')


    def is_adopted(node):
        tree = node.node_tree
        if tree is None:return False
        current = _records.get((tree.as_pointer(), node.as_pointer()))
        if current is None or current['epoch'] != generation_key():return False
        try:
            if resolve(current['tree']) != tree or resolve(current['owner']) != node:return False
        except RuntimeError:
            # A fenced query cannot reacquire RNA. Existing adoption still only
            # suppresses writes; it grants neither numerical nor native authority.
            return bool(current['state'].adopted)
        return bool(current['state'].adopted)


    def has_runtime_intent(node):
        tree = node.node_tree
        if tree is None:return False
        current = _records.get((tree.as_pointer(), node.as_pointer()))
        if (current is None or current['epoch'] != generation_key()
                or resolve(current['tree']) != tree or resolve(current['owner']) != node):
            return False
        state = current['state']
        return not state.adopted and (state.intent != 0 or state.operation > 0)


    def request_intent(node, intent, explicit=False):
        current = record(node.node_tree, node)
        state = current['state']
        _, accepted = dispatch(current, 'intent', session=state.session,
            incarnation=state.incarnation, intent=json.dumps(intent, sort_keys=True,
            separators=(',', ':')), explicit=bool(explicit))
        return accepted


    def record(tree, owner=None):
        # Native banks and constants belong to a private writable kernel. Shared
        # restored kernels are read-only; each owner has its own runtime receipt.
        if owner is None:
            matches = [value for value in _records.values()
                       if value['epoch'] == generation_key() and resolve(value['tree']) == tree]
            if len(matches) == 1:
                return matches[0]
            raise ValueError('Output authority requires its owning node')
        key = (tree.as_pointer(), owner.as_pointer())
        current = _records.get(key)
        if (current is None or current['epoch'] != generation_key()
                or resolve(current['tree']) != tree or resolve(current['owner']) != owner):
            if current is not None and _repair_pending(current):
                if current.get('kernel_uid') != int(tree.session_uid):
                    raise RuntimeError('LUT repair target changed without native replacement receipt')
                # Manual recovery invalidates Python borrows, not the native
                # transaction. Keep its exact actor and value-only repair plan.
                current.update(tree=retain(tree, 'output_resources.kernel'),
                               owner=retain(owner, 'output_resources.owner'),
                               epoch=generation_key(), repair_rebind=True)
                return current
            if current is None:
                store = _application.app.store(machine=_application.machines['output'],
                    initial=dict(incarnation=next(_serial), uncertain=True, session=_session))
            else:
                dispatch(current, 'invalidate', incarnation=current['store'].state.incarnation)
                store = current['store']
            current = ResourceRecord(tree=retain(tree, 'output_resources.kernel'), epoch=generation_key(),
                                     owner=retain(owner, 'output_resources.owner'), store=store,
                                     kernel_uid=int(tree.session_uid))
            _records[key] = current
        return current


    def selected_index(node):
        if not is_current(node):
            return -1
        value = float(selector_socket(node).default_value)
        return int(value) if value in (0., 1., 2., 3.) else -1


    def bank_images(node, snapshot=None):
        """One bounded lookup; all wrappers die with its synchronous caller."""
        items = ((item, item.image) for item in node.node_tree.nodes.values()
                 if item.get('_pg_cube_image')) if snapshot is None else snapshot[2]
        images = {}
        for item, image in items:
            index = item.get(SLOT)
            if type(index) is not int or index not in range(4) or image is None:
                raise ValueError('Incomplete permanent LUT bank')
            if index in images and images[index] != image:
                raise ValueError('Mixed Images within a permanent LUT bank')
            images[index] = image
        if set(images) != {0, 1, 2, 3}:
            raise ValueError('Missing permanent LUT bank')
        if len({image.as_pointer() for image in images.values()}) != 4:
            raise ValueError('Writable LUT banks must have distinct Images')
        return images


    def current_image(node, snapshot=None):
        index = selected_index(node)
        if index < 0:
            return None
        store = record(node.node_tree, node)['store']
        state = store.state
        slot = store.query('slot_for', index=index)
        if state.selected != index or not state.graph_valid or not store.query('complete', index=index, source=slot.source):
            return None
        image = bank_images(node, snapshot)[index]
        if image.get('_pg_cube_signature') != slot.source:
            return None
        return image


    def _trees():
        trees = list(bpy.data.node_groups)
        for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights, bpy.data.scenes):
            trees.extend(item.node_tree for item in collection if getattr(item, 'node_tree', None))
        return trees


    def unique_owner(node):
        """Count actual owners, rather than relying on native transient user counts."""
        kernel = node.node_tree
        if node.id_data.is_evaluated or node.id_data.library or kernel.library:
            return False
        owners = 0
        for tree in _trees():
            if tree.is_evaluated:
                continue
            for item in tree.nodes:
                if getattr(item, 'node_tree', None) == kernel:
                    owners += 1
                    if owners > 1:
                        return False
        return owners == 1


    def writable_banks(node):
        """Independent kernels can still alias the same writable Image."""
        images = bank_images(node)
        if any(image.library or image.is_evaluated for image in images.values()):
            return False
        pointers = {image.as_pointer() for image in images.values()}
        for tree in _trees():
            if tree == node.node_tree or tree.is_evaluated:
                continue
            for item in tree.nodes:
                image = getattr(item, 'image', None)
                if image is not None and image.as_pointer() in pointers:
                    return False
        return True


    def _step(node, event, **values):
        output_guard = _application.component('graph.output_guard')
        current = record(node.node_tree, node)
        state = current['state']
        arguments = dict(session=state.session, incarnation=state.incarnation,
                         reader_excluded=output_guard.publishing())
        arguments.update(values)
        state, accepted = dispatch(current, event, **arguments)
        if not accepted:
            raise RuntimeError('Stale or incomplete LUT resource receipt: ' + event)
        return state


    def reserve(node, size, source):
        """Called while the host reader lease is held; debt precedes the write."""
        output_guard = _application.component('graph.output_guard')
        if not output_guard.publishing():
            raise RuntimeError('LUT reservation requires native reader exclusion')
        if not unique_owner(node):
            raise RuntimeError('Use Make Independent before updating shared LUT resources')
        if not writable_banks(node):
            raise RuntimeError('Use Make Independent before updating linked or evaluated LUT banks')
        current = record(node.node_tree, node)
        # Native selection protects storage independently of numerical provenance.
        _step(node, 'protect', slot=selected_index(node))
        state = current['state']
        candidates = (0, 1) if size == 33 else (2, 3)
        index = next((index for index in candidates
                      if index != state.selected and index != state.protected), None)
        if index is None:raise RuntimeError('No unselected LUT bank is available')
        state, accepted = dispatch(current, 'reserve', slot=index,
            session=state.session, incarnation=state.incarnation, source=source, reader_excluded=True)
        if not accepted:
            raise RuntimeError('LUT storage reservation was rejected')
        operation_id = uuid.uuid4().hex
        current['operation_id'] = operation_id
        return dict(index=index, session=state.session, incarnation=state.incarnation, source=source,
                    operation=state.operation, store=current['store'].identity, operation_id=operation_id)


    def receipt(node, event, reservation, validated=False, *, clean=False):
        output_guard = _application.component('graph.output_guard')
        current = record(node.node_tree, node)
        _require_token_owner(current, reservation)
        state, accepted = dispatch(current, event,
            slot=reservation['index'], incarnation=reservation['incarnation'],
            session=reservation['session'],
            source=reservation['source'], operation=reservation['operation'],
            validated=validated, clean=clean, reader_excluded=output_guard.publishing())
        if not accepted:
            raise RuntimeError('Stale or incomplete LUT payload receipt: ' + event)


    def graph_begin(node, index, source, output_key):
        if not unique_owner(node):
            raise RuntimeError('Native output constants require private writable resources')
        state = _step(node, 'graph_begin', slot=index, source=source, output_key=output_key)
        return dict(index=index, source=source, output_key=output_key,
                    session=state.session, incarnation=state.incarnation,
                    operation=state.graph_operation,
                    store=record(node.node_tree, node)['store'].identity)


    def graph_receipt(node, event, token, validated=False):
        _require_token_owner(record(node.node_tree, node), token)
        values = dict(token)
        values.pop('store')
        values['slot'] = values.pop('index')
        return _step(node, event, validated=validated, **values)


    def select_payload(node, index):
        """Scalar-only selection. Graph relationships are established at creation."""
        if index not in (-1, 0, 1, 2, 3):
            raise ValueError('Invalid LUT output selector')
        # Reassigning an unchanged RNA socket still dirties the owning node tree
        # and emits a dependency-graph update.  Settled cache hits call this path
        # from every idle poll, so make selection idempotent at the native boundary.
        value = float(index)
        socket = selector_socket(node)
        if socket.default_value != value:
            socket.default_value = value


    def pixel_digest(pixels):
        return hashlib.sha256(memoryview(np.ascontiguousarray(pixels)).cast('B')).hexdigest()


    def payload_record(image):
        """Persisted producer history never grants current numerical authority."""
        try:
            value = json.loads(image.get(RECORD, ''))
        except (TypeError, ValueError):
            return None
        return value if (isinstance(value, dict) and type(value.get('version')) is int
                         and value['version'] == RECORD_VERSION) else None


    def _description(image):
        lut = _application.component('graph.lut')
        from ...perceptual_grade_core.model import lut as cm
        return dict(version=RECORD_VERSION, algorithm=cm.SCHEMA, layout=LAYOUT,
                    cube_schema=lut.SCHEMA, size=image.get('_pg_cube_size'),
                    encoding=image.get('_pg_cube_encoding'), byte_count=len(image.pixels) * 4)


    def _write_record(image, phase, *, source='', revision=0, reservation=None):
        value = _description(image)
        value.update(phase=phase, source=source, revision=revision)
        if reservation is not None:
            value.update((key, reservation[key]) for key in ('session', 'incarnation', 'operation'))
            value['operation_id'] = reservation.get('operation_id', '')
        image[RECORD] = json.dumps(value, sort_keys=True, separators=(',', ':'))
        if payload_record(image) != value:
            raise RuntimeError('LUT write record did not persist in native data')
        return value


    def validate_layout(node, images=None):
        """Structural mismatch requires an explicit lifecycle operation."""
        lut = _application.component('graph.lut')
        from ...perceptual_grade_core.model import lut as cm
        images = bank_images(node) if images is None else images
        encoding = 'RGBE_DIFFERENCES' if node.node_tree.bl_idname == 'CompositorNodeTree' else 'RGB'
        for index, image in images.items():
            size = cm.PREVIEW_SIZE if index < 2 else cm.SIZE
            expected = (cm.ATLAS_WIDTH, cm.atlas_height(size, 4 if encoding == 'RGBE_DIFFERENCES' else 1))
            if (image.get('_pg_cube_schema') != lut.SCHEMA or image.get('_pg_cube_size') != size
                    or image.get('_pg_cube_encoding') != encoding or tuple(image.size) != expected
                    or len(image.pixels) != expected[0] * expected[1] * 4):
                raise ValueError('Invalid LUT bank layout')
        return images


    def read_pixels(image):
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        return validate_pixels(image, values)


    def pack_image(image):
        """Pack a dirty, privately admitted Image and check native success.

        Blender's memory pack can fail without a Python exception. Its clean-image
        no-op is not evidence that restored packed bytes match retained live pixels.
        Callers establish structural admission or hold the publication lease.
        """
        if image.library is not None or image.is_evaluated:
            raise RuntimeError('Cannot pack linked or evaluated LUT storage')
        image.pack()
        packed = image.packed_file
        if packed is None or packed.size <= 0 or image.is_dirty:
            raise RuntimeError('LUT image packing did not complete')


    def observe_packing(node, index, image=None):
        """Observe warning eligibility without writing or certifying packed bytes."""
        target = bank_images(node)[index]
        if image is not None and image != target:
            raise RuntimeError('LUT packing observation target changed')
        state = record(node.node_tree, node)['state']
        _step(node, 'observe_packing', slot=index, operation=state.operation,
              validated=True, clean=not target.is_dirty)


    def ensure_packed(node, index, reservation=None):
        """Discharge only this bank's packing work, without a pixel rewrite.

        Unowned clean banks carry no new packed-byte correspondence. Every newly
        written bank requires a checked, nonempty packed file.
        """
        output_guard = _application.component('graph.output_guard')
        if not output_guard.publishing():
            raise RuntimeError('LUT packing requires a native resource lease')
        image = bank_images(node)[index]
        current = record(node.node_tree, node)
        state = current['state']
        token = (dict(reservation) if reservation is not None else
                 dict(index=index, session=state.session, incarnation=state.incarnation,
                      operation=state.operation, source=state.slots[index].source,
                      store=current['store'].identity))
        _require_token_owner(current, token)
        if (token['index'] != index or token['session'] != state.session
                or token['incarnation'] != state.incarnation or token['operation'] != state.operation
                or (reservation is not None and (state.pending != index
                    or state.pending_source != token['source'] or state.write_phase != policy.WRITE_ARMED))
                or (reservation is None and state.pending != -1)):
            raise RuntimeError('LUT packing requires a current resource token')
        if reservation is None and not image.is_dirty:
            observe_packing(node, index, image)
            return
        receipt(node, 'pack_failed', token)
        try:
            if not unique_owner(node) or not writable_banks(node):
                raise RuntimeError('LUT packing requires private writable resources')
            pack_image(image)
            receipt(node, 'packed', token, validated=True, clean=True)
        except Exception:
            receipt(node, 'pack_failed', token)
            raise


    def arm(node, reservation, image):
        """Install the serialized intention before permitting any buffer write."""
        current = record(node.node_tree, node)
        _require_token_owner(current, reservation)
        if (type(reservation.get('operation_id')) is not str
                or reservation['operation_id'] != current.get('operation_id')):
            raise RuntimeError('LUT write intention requires its producer operation identity')
        if (type(reservation.get('index')) is not int or reservation['index'] not in range(4)
                or bank_images(node)[reservation['index']] != image):
            raise RuntimeError('LUT write intention target differs from its reservation')
        _write_record(image, 'WRITING', source=reservation['source'], reservation=reservation)
        receipt(node, 'armed', reservation, validated=True)


    def validate_pixels(image, pixels):
        """Also bounds intermediate samples from a partially overwritten bank."""
        values = np.asarray(pixels)
        if values.dtype != np.float32 or values.size != len(image.pixels):
            raise ValueError('Invalid permanent LUT payload dimensions or dtype')
        # Reductions inspect every float without allocating full-atlas masks or
        # absolute-value arrays. NaNs propagate to the extrema and fail these
        # comparisons; infinities exceed the same finite bounds.
        if image.get('_pg_cube_encoding') == 'RGBE_DIFFERENCES':
            rgba = values.reshape(-1, 4)
            rgb, exponent = rgba[:, :3], rgba[:, 3]
            rgb_min, rgb_max = rgb.min(initial=0), rgb.max(initial=0)
            exponent_min, exponent_max = exponent.min(initial=0), exponent.max(initial=0)
            # These channel extrema already cover the entire atlas. Reuse them
            # for the general bounds instead of scanning every float twice more.
            if not (rgb_min >= -2.**120 and rgb_max <= 2.**120
                    and exponent_min >= -2.**120 and exponent_max <= 2.**120):
                raise ValueError('LUT payload is not finite bounded float32 data')
            if not (rgb_min >= -1. and rgb_max <= 1.
                    and exponent_min >= -149 and exponent_max <= 120
                    and np.equal(exponent, np.floor(exponent)).all()):
                raise ValueError('Invalid bounded RGBE payload')
        elif not (values.min(initial=0) >= -2.**120 and values.max(initial=0) <= 2.**120):
            raise ValueError('LUT payload is not finite bounded float32 data')
        return np.ascontiguousarray(values).ravel()


    def transfer_payload(image, pixels, cube, report, *, node, reservation):
        """Complete one fenced producer through ordered successful native effects."""
        output_guard = _application.component('graph.output_guard')
        if not output_guard.publishing():
            raise RuntimeError('LUT transfer requires native reader exclusion')
        current = record(node.node_tree, node)
        _require_token_owner(current, reservation)
        if (bank_images(node)[reservation['index']] != image
                or cube.signature != reservation['source']):
            raise RuntimeError('LUT transfer target or source differs from its reservation')
        state = current['state']
        if (state.session != reservation['session'] or state.incarnation != reservation['incarnation']
                or state.pending != reservation['index'] or state.operation != reservation['operation']
                or state.pending_source != reservation['source'] or state.write_phase != policy.WRITE_ARMED):
            raise RuntimeError('LUT transfer requires a current armed reservation')
        saved = payload_record(image)
        if (saved is None or saved.get('phase') != 'WRITING'
                or any(saved.get(key) != reservation[key] for key in ('session', 'incarnation', 'operation', 'source', 'operation_id'))):
            raise RuntimeError('LUT transfer requires a matching serialized write intention')
        values = validate_pixels(image, pixels)
        image.pixels.foreach_set(values)
        image.update()
        _require_token_owner(record(node.node_tree, node), reservation)
        image['_pg_cube_signature'] = cube.signature
        image[REVISION] = reservation['operation_id']
        for key, value in report.items():
            if type(value) in (str, bool, int, float):
                image['_pg_bake_' + key] = value
        ensure_packed(node, reservation['index'], reservation)
        image.update_tag()
        _require_token_owner(record(node.node_tree, node), reservation)
        _write_record(image, 'COMPLETE', source=cube.signature,
                      revision=image[REVISION], reservation=reservation)


    def pack_output(node):
        """Packing belongs to the producer; saving preserves current native data."""
        return None


    def _repair_pending(current):
        protocol = current.get('publication_protocol')
        return (protocol is not None and protocol.state.debt or
                any(current.get(key) is not None for key in
                    ('publication_repair', 'repair_selection')))


    def native_replaced():
        """Retire old obligations only after an actual native pre/post boundary."""
        for current in tuple(_records.values()):
            protocol = current.get('publication_protocol')
            if protocol is not None and not protocol.store.ref.closed:
                if protocol.state.debt:
                    protocol.require('superseded', 'release',
                        generation=protocol.state.generation + 1, validated=True)
                protocol.event('close')
                protocol.close()
            if not current['store'].ref.closed:current['store'].close()
        _records.clear()


    def clear():
        if any(_repair_pending(current) for current in _records.values()):
            raise RuntimeError('LUT publication repair must finish before clearing resources')
        for current in tuple(_records.values()):
            protocol = current.get('publication_protocol')
            if protocol is not None and not protocol.store.ref.closed:
                protocol.event('close')
                protocol.close()
            if not current['store'].ref.closed:current['store'].close()
        _records.clear()

    def _set_export_has_runtime_intent(value):
        nonlocal has_runtime_intent
        has_runtime_intent = value

    def _set_export_adopt_native(value):
        nonlocal adopt_native
        adopt_native = value

    def _set_export_is_adopted(value):
        nonlocal is_adopted
        is_adopted = value

    def _set_export_request_intent(value):
        nonlocal request_intent
        request_intent = value

    def _set_export_LAYOUT(value):
        nonlocal LAYOUT
        LAYOUT = value

    def _set_export_LAYOUT_KEY(value):
        nonlocal LAYOUT_KEY
        LAYOUT_KEY = value

    def _set_export_RECORD(value):
        nonlocal RECORD
        RECORD = value

    def _set_export_RECORD_VERSION(value):
        nonlocal RECORD_VERSION
        RECORD_VERSION = value

    def _set_export_REVISION(value):
        nonlocal REVISION
        REVISION = value

    def _set_export_SELECTOR(value):
        nonlocal SELECTOR
        SELECTOR = value

    def _set_export_SLOT(value):
        nonlocal SLOT
        SLOT = value

    def _set_export__records(value):
        nonlocal _records
        _records = value

    def _set_export__serial(value):
        nonlocal _serial
        _serial = value

    def _set_export__session(value):
        nonlocal _session
        _session = value

    def _set_export_ResourceRecord(value):
        nonlocal ResourceRecord
        ResourceRecord = value

    def _set_export_dispatch(value):
        nonlocal dispatch
        dispatch = value

    def _set_export_is_current(value):
        nonlocal is_current
        is_current = value

    def _set_export_record(value):
        nonlocal record
        record = value

    def _set_export_selected_index(value):
        nonlocal selected_index
        selected_index = value

    def _set_export_bank_images(value):
        nonlocal bank_images
        bank_images = value

    def _set_export_current_image(value):
        nonlocal current_image
        current_image = value

    def _set_export__trees(value):
        nonlocal _trees
        _trees = value

    def _set_export_unique_owner(value):
        nonlocal unique_owner
        unique_owner = value

    def _set_export_writable_banks(value):
        nonlocal writable_banks
        writable_banks = value

    def _set_export__step(value):
        nonlocal _step
        _step = value

    def _set_export_reserve(value):
        nonlocal reserve
        reserve = value

    def _set_export_receipt(value):
        nonlocal receipt
        receipt = value

    def _set_export_graph_begin(value):
        nonlocal graph_begin
        graph_begin = value

    def _set_export_graph_receipt(value):
        nonlocal graph_receipt
        graph_receipt = value

    def _set_export_select_payload(value):
        nonlocal select_payload
        select_payload = value

    def _set_export_pixel_digest(value):
        nonlocal pixel_digest
        pixel_digest = value

    def _set_export_payload_record(value):
        nonlocal payload_record
        payload_record = value

    def _set_export__description(value):
        nonlocal _description
        _description = value

    def _set_export__write_record(value):
        nonlocal _write_record
        _write_record = value

    def _set_export_validate_layout(value):
        nonlocal validate_layout
        validate_layout = value

    def _set_export_read_pixels(value):
        nonlocal read_pixels
        read_pixels = value

    def _set_export_pack_image(value):
        nonlocal pack_image
        pack_image = value

    def _set_export_observe_packing(value):
        nonlocal observe_packing
        observe_packing = value

    def _set_export_ensure_packed(value):
        nonlocal ensure_packed
        ensure_packed = value

    def _set_export_arm(value):
        nonlocal arm
        arm = value

    def _set_export_validate_pixels(value):
        nonlocal validate_pixels
        validate_pixels = value

    def _set_export_transfer_payload(value):
        nonlocal transfer_payload
        transfer_payload = value

    def _set_export_pack_output(value):
        nonlocal pack_output
        pack_output = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'adopt_native': lambda: adopt_native,
        'is_adopted': lambda: is_adopted,
        'has_runtime_intent': lambda: has_runtime_intent,
        'request_intent': lambda: request_intent,
        'LAYOUT': lambda: LAYOUT,
        'LAYOUT_KEY': lambda: LAYOUT_KEY,
        'RECORD': lambda: RECORD,
        'RECORD_VERSION': lambda: RECORD_VERSION,
        'REVISION': lambda: REVISION,
        'ResourceRecord': lambda: ResourceRecord,
        'SELECTOR': lambda: SELECTOR,
        'create_selector': lambda: create_selector,
        'selector_socket': lambda: selector_socket,
        'SLOT': lambda: SLOT,
        '_description': lambda: _description,
        '_records': lambda: _records,
        '_serial': lambda: _serial,
        '_session': lambda: _session,
        '_step': lambda: _step,
        '_trees': lambda: _trees,
        '_write_record': lambda: _write_record,
        'arm': lambda: arm,
        'bank_images': lambda: bank_images,
        'bpy': lambda: bpy,
        'clear': lambda: clear,
        'native_replaced': lambda: native_replaced,
        'count': lambda: count,
        'current_image': lambda: current_image,
        'dispatch': lambda: dispatch,
        'ensure_packed': lambda: ensure_packed,
        'generation_key': lambda: generation_key,
        'graph_begin': lambda: graph_begin,
        'graph_receipt': lambda: graph_receipt,
        'hashlib': lambda: hashlib,
        'is_current': lambda: is_current,
        'json': lambda: json,
        'np': lambda: np,
        'observe_packing': lambda: observe_packing,
        'pack_image': lambda: pack_image,
        'pack_output': lambda: pack_output,
        'payload_record': lambda: payload_record,
        'pixel_digest': lambda: pixel_digest,
        'policy': lambda: policy,
        'read_pixels': lambda: read_pixels,
        'receipt': lambda: receipt,
        'record': lambda: record,
        'reserve': lambda: reserve,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'select_payload': lambda: select_payload,
        'selected_index': lambda: selected_index,
        'transfer_payload': lambda: transfer_payload,
        'unique_owner': lambda: unique_owner,
        'uuid': lambda: uuid,
        'validate_layout': lambda: validate_layout,
        'validate_pixels': lambda: validate_pixels,
        'writable_banks': lambda: writable_banks,
    }, writers={'has_runtime_intent': _set_export_has_runtime_intent, 'adopt_native': _set_export_adopt_native, 'is_adopted': _set_export_is_adopted, 'request_intent': _set_export_request_intent, 
        'LAYOUT': _set_export_LAYOUT,
        'LAYOUT_KEY': _set_export_LAYOUT_KEY,
        'RECORD': _set_export_RECORD,
        'RECORD_VERSION': _set_export_RECORD_VERSION,
        'REVISION': _set_export_REVISION,
        'SELECTOR': _set_export_SELECTOR,
        'SLOT': _set_export_SLOT,
        '_records': _set_export__records,
        '_serial': _set_export__serial,
        '_session': _set_export__session,
    'ResourceRecord': _set_export_ResourceRecord, 'dispatch': _set_export_dispatch, 'is_current': _set_export_is_current, 'record': _set_export_record, 'selected_index': _set_export_selected_index, 'bank_images': _set_export_bank_images, 'current_image': _set_export_current_image, '_trees': _set_export__trees, 'unique_owner': _set_export_unique_owner, 'writable_banks': _set_export_writable_banks, '_step': _set_export__step, 'reserve': _set_export_reserve, 'receipt': _set_export_receipt, 'graph_begin': _set_export_graph_begin, 'graph_receipt': _set_export_graph_receipt, 'select_payload': _set_export_select_payload, 'pixel_digest': _set_export_pixel_digest, 'payload_record': _set_export_payload_record, '_description': _set_export__description, '_write_record': _set_export__write_record, 'validate_layout': _set_export_validate_layout, 'read_pixels': _set_export_read_pixels, 'pack_image': _set_export_pack_image, 'observe_packing': _set_export_observe_packing, 'ensure_packed': _set_export_ensure_packed, 'arm': _set_export_arm, 'validate_pixels': _set_export_validate_pixels, 'transfer_payload': _set_export_transfer_payload, 'pack_output': _set_export_pack_output, 'clear': _set_export_clear})
