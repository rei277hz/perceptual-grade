"""Compensation ledgers for native datablock edits.

Only a live, synchronous Transaction contains callbacks. Failed repairs retain
copied scalar identities and values, never Blender wrappers or exceptions.
"""

from ..integration.components import Component

def create(_application):
    from itertools import count
    from uuid import uuid4
    from ...blender_runtime.execution.diagnostics import text
    import json

    SCHEMA = 1


    def plain(value):
        """Reject an RNA object or exception before it can enter persistent debt."""
        if value is None or type(value) in (str, int, float, bool): return value
        if type(value) in (tuple, list): return tuple(plain(item) for item in value)
        if type(value) is dict:
            if not all(type(key) is str for key in value):
                raise TypeError('Repair record keys must be strings')
            return {key: plain(item) for key, item in value.items()}
        raise TypeError('Repair records must contain only copied values')


    _pending = {}
    _occurrences = count(1)
    _session = _application.app.identity
    _discovered = None
    RECORDS = '_pg_ownership_repairs'
    _edge_queries = _application.app.store(machine=_application.machines['edge_effect'],
        initial=dict(occurrence=1, before=0, after=0))


    def edge_token(value):
        """Lossless scalar token for the immutable native-identity tuple domain."""
        if value is None:
            encoded = ['none']
        elif type(value) is int:
            encoded = ['int', str(value)]
        elif type(value) is str:
            encoded = ['str', value]
        elif type(value) is tuple:
            encoded = ['tuple', [edge_token(item) for item in value]]
        else:
            raise TypeError('Native edge identity must contain only tuples, strings, integers or None')
        return json.dumps(encoded, ensure_ascii=True, separators=(',', ':'))


    def edge_count(edges, selected):
        return _edge_queries.query('edge_count', edges=tuple(edge_token(edge) for edge in edges),
                                   selected=edge_token(selected))


    def effect_event(store, kind, occurrence):
        event = store.machine.event_type(kind=kind, occurrence=occurrence)
        store.dispatch(event)
        return store.state


    class Transaction:
        def __init__(self, kind, record):
            self.app = _application.app
            self.arena = self.app.ownership
            self.identity = uuid4().hex
            self.generation = self.arena.generation_key()
            self.kind = str(kind)
            self.record = plain(record)
            self.effects = []
            self.closed = False
            self.committed = False

        def check_generation(self):
            self.app.require_open()
            if self.arena.generation_key() != self.generation:
                raise RuntimeError('Native ownership transaction crossed a replacement boundary')

        def persist(self):
            self.check_generation()
            debt = plain({'kind': self.kind, 'record': self.record,
                          'committed': self.committed, 'failures': (),
                          'session': _session, 'generation': self.generation})
            _pending[self.identity] = debt
            write_debt(self.identity, debt)

        def apply(self, name, action, compensate, before_edges=(), after_edges=(), retirement=False):
            occurrence = next(_occurrences)
            before_edges, after_edges = tuple(before_edges), tuple(after_edges)
            # The set selects identities; edge_count retains occurrence multiplicity.
            identities = set(before_edges + after_edges) or {None}
            stores = [self.app.store(machine=_application.machines['edge_effect'], initial=dict(
                        occurrence=occurrence, before=edge_count(before_edges, identity),
                        after=edge_count(after_edges, identity))) for identity in identities]
            states = [effect_event(store, 'ARM', occurrence) for store in stores]
            effect = {'name': str(name), 'occurrence': occurrence, 'states': states, 'stores': stores,
                      'compensate': compensate, 'retirement': bool(retirement)}
            self.effects.append(effect)  # Before any native mutation, including partial writes.
            self.check_generation()
            self.persist()  # Durable intent precedes the first and every later effect.
            result = action()
            self.check_generation()
            # Native action/undo callbacks may have re-entered; reduce the receipt
            # against each Store's current state, never the pre-action state list.
            effect['states'] = [effect_event(store, 'APPLY', occurrence) for store in stores]
            return result

        def commit(self):
            self.check_generation()
            for effect in self.effects:
                effect['states'] = [effect_event(store, 'COMMIT', effect['occurrence'])
                                    for store in effect['stores']]
            if any(not store.state.retired for effect in self.effects for store in effect['stores']):
                raise RuntimeError('Cannot commit an incomplete native transaction')
            self.committed = True
            if self.effects:
                self.persist()
                erase_debt(self.identity, self.record)

        def rollback(self):
            if self.committed:
                return False  # Native commit must never roll back through a removed owner.
            if self.arena.generation_key() != self.generation:
                return False  # Replacement owns retirement; never rebase compensation.
            failures = []
            for effect in reversed(self.effects):
                try:
                    self.check_generation()
                    if effect['retirement'] and failures:
                        raise RuntimeError('Native retirement waits for the remaining compensations')
                    effect['compensate']()
                    self.check_generation()
                except Exception as error:
                    if self.arena.generation_key() != self.generation:
                        return False
                    failures.append((effect['name'], text(type(error).__name__), text(error)))
                    event = 'COMPENSATION_FAILED'
                else:
                    event = 'COMPENSATED'
                effect['states'] = [effect_event(store, event, effect['occurrence'])
                                    for store in effect['stores']]
            if failures:
                debt = plain({'kind': self.kind, 'record': self.record,
                              'committed': False, 'failures': failures,
                              'session': _session, 'generation': self.generation})
                _pending[self.identity] = debt
                try: write_debt(self.identity, debt)
                except Exception: pass  # The previously written intent remains dirty.
            else:
                try: erase_debt(self.identity, self.record)
                except Exception as error:
                    _pending[self.identity] = plain({'kind': self.kind, 'record': self.record,
                        'committed': False, 'failures': (('journal', text(type(error).__name__), text(error)),),
                        'session': _session, 'generation': self.generation})
                    failures.append(('journal', 'RuntimeError', 'Could not retire repair intent'))
            return not failures

        def close(self):
            # Also runs when the raised exception retains this transaction's frame.
            for effect in self.effects:
                effect.pop('compensate', None)
            self.effects.clear()
            self.closed = True


    def pending(*, cleanup=False):
        if cleanup:
            # Terminal cleanup cannot reopen a failed arena merely to discover
            # work. Read the existing ledgers without rebasing native identities.
            if not _application.closing or _application.waiting_for_readers:
                raise RuntimeError('Cleanup debt inspection requires the original guarded teardown')
            if _pending:
                return True
            return False
        arena = _application.app.ownership
        if _discovered != arena.generation_key():
            if not arena.admits('rebind'): return True
            available = native_available()
            if available is None: return bool(_pending)
            if not available: return True
            discover()
        return bool(_pending)


    def require_clean():
        if pending():
            raise RuntimeError('Datablock ownership repair is still pending')


    def repair_all():
        grades = _application.component('graph.grades'); stacks = _application.component('graph.stacks')
        discover()
        for identity, debt in tuple(_pending.items()):
            try:
                if (debt.get('session') != _session or
                        tuple(debt.get('generation', ())) != _application.app.ownership.generation_key()):
                    raise RuntimeError('Native replacement must retire the old repair operation')
                if debt.get('committed'):
                    if debt['kind'] == 'replacement': grades.verify_replacement(debt['record'])
                    elif debt['kind'] == 'stack': stacks.verify_assignment(debt['record'], after=True)
                    elif debt['kind'] == 'kernel': verify_kernel(debt['record'], after=True)
                    elif debt['kind'] == 'driver': verify_driver(debt['record'], after=True)
                    elif debt['kind'] == 'grade_create': grades.verify_grade_creation(debt['record'], after=True)
                    elif debt['kind'] == 'bake_binding': _application.component('graph.bake').verify_binding(debt['record'], after=True)
                    elif debt['kind'] == 'bake_create': _application.component('graph.bake').verify_image_creation(debt['record'], after=True)
                    else: raise RuntimeError('Unknown committed datablock repair kind')
                elif debt['kind'] == 'replacement': grades.repair_replacement(debt['record'])
                elif debt['kind'] == 'stack': stacks.repair_assignment(debt['record'])
                elif debt['kind'] == 'kernel': repair_kernel(debt['record'])
                elif debt['kind'] == 'driver': repair_driver(debt['record'])
                elif debt['kind'] == 'grade_create': grades.repair_grade_creation(debt['record'])
                elif debt['kind'] == 'bake_binding': _application.component('graph.bake').repair_binding(debt['record'])
                elif debt['kind'] == 'bake_create': _application.component('graph.bake').repair_image_creation(debt['record'])
                else: raise RuntimeError('Unknown datablock repair kind')
                erase_debt(identity, debt['record'])
            except Exception as error:
                _pending[identity] = plain({**debt,
                    'failures': ((debt['kind'], text(type(error).__name__), text(error)),)})
        require_clean()


    def id_key(value):
        if value is None: return None
        arena = _application.app.ownership
        library = value.library.filepath if value.library else ''
        if not library or value.override_library:
            if NODE_ID not in value: value[NODE_ID] = uuid4().hex
        return ('ID', _session, tuple(arena.generation_key()), int(value.session_uid),
                str(value.get(NODE_ID, '')), value.name_full, library, id_kind(value))


    def id_kind(value):
        return getattr(value, 'bl_idname', value.bl_rna.identifier)


    def fresh_identity(value):
        value[NODE_ID] = uuid4().hex
        return id_key(value)


    def trees():
        import bpy
        yield from bpy.data.node_groups
        for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights, bpy.data.scenes):
            for owner in collection:
                tree = getattr(owner, 'node_tree', None)
                if tree is not None: yield tree


    def find_tree(identity):
        return find_id(identity, trees())


    def find_action(identity):
        import bpy
        return find_id(identity, bpy.data.actions)


    def find_id(identity, values):
        if identity is None: return None
        arena = _application.app.ownership
        if identity[0] != 'ID': raise ValueError('Unknown datablock identity format')
        current = identity[1] == _session and tuple(identity[2]) == arena.generation_key()
        if current:
            return next((value for value in values if int(value.session_uid) == identity[3]), None)
        candidates = []
        for value in values:
            library = value.library.filepath if value.library else ''
            if library != identity[6] or id_kind(value) != identity[7]: continue
            if identity[4] and value.get(NODE_ID) != identity[4]: continue
            if value.name_full == identity[5]: return value
            if identity[4]: candidates.append(value)
        if len(candidates) == 1: return candidates[0]
        if candidates: raise RuntimeError('Saved datablock identity is ambiguous')
        return None


    def require_tree(identity):
        tree = find_tree(identity)
        if tree is None: raise RuntimeError('A referenced datablock no longer exists')
        return tree


    NODE_ID = '_pg_ownership_identity'


    def node_key(node):
        arena = _application.app.ownership
        if NODE_ID not in node: node[NODE_ID] = uuid4().hex
        return {'tree': id_key(node.id_data), 'pointer': int(node.as_pointer()),
                'identity': str(node[NODE_ID]), 'name': node.name,
                'session': _session, 'generation': tuple(arena.generation_key())}


    def find_node(tree, identity):
        if identity is None: return None
        arena = _application.app.ownership
        current = (identity['session'] == _session and
                   tuple(identity['generation']) == arena.generation_key() and
                   int(tree.session_uid) == identity['tree'][3])
        candidates = [node for node in tree.nodes if node.get(NODE_ID) == identity['identity']
                      and (not current or node.as_pointer() == identity['pointer'])]
        if len(candidates) == 1: return candidates[0]
        match = next((node for node in candidates if node.name == identity['name']), None)
        if match is None and candidates:
            raise RuntimeError('Saved node identity is ambiguous')
        return match


    def owner_key(record):
        return record['node']['tree'] if 'node' in record else record['tree']


    def read_debts(tree):
        raw = tree.get(RECORDS)
        if raw is None: return {}
        payload = json.loads(raw)
        if payload.get('schema') != SCHEMA or type(payload.get('debts')) is not dict:
            raise RuntimeError('Unsupported datablock repair snapshot')
        return plain(payload['debts'])


    def write_debt(identity, debt):
        tree = require_tree(owner_key(debt['record']))
        values = read_debts(tree)
        values[identity] = plain(debt)
        tree[RECORDS] = json.dumps({'schema': SCHEMA, 'debts': values}, separators=(',', ':'))


    def erase_debt(identity, record):
        tree = find_tree(owner_key(record))
        if tree is not None:
            values = read_debts(tree)
            values.pop(identity, None)
            if values:
                tree[RECORDS] = json.dumps({'schema': SCHEMA, 'debts': values}, separators=(',', ':'))
            elif RECORDS in tree:
                del tree[RECORDS]
        _pending.pop(identity, None)


    def rebase_record(value, before, after):
        if type(value) is dict:
            result = {key: rebase_record(item, before, after) for key, item in value.items()}
            if 'pointer' in result and 'generation' in result and tuple(before) != tuple(after):
                result['session'] = ''  # Copied/reloaded nodes require tree-scoped UUID resolution.
            return result
        if type(value) in (tuple, list):
            if tuple(value) == tuple(before): return after
            return tuple(rebase_record(item, before, after) for item in value)
        return value


    def discover():
        """Bind runtime discovery without importing historical commands.

        A saved repair record describes an old operation. Loading or restoring
        that record cannot authorize compensation against the current native
        graph. Only this Session's live transactions populate _pending.
        """
        nonlocal _discovered
        arena = _application.app.ownership
        if not native_available(): return
        _discovered = arena.generation_key()


    def native_replaced(tokens):
        """Retire compensation authority after an actual native replacement.

        The recovery coordinator owns this effect. Its receipt settles runtime
        bookkeeping only: historical native journal properties stay untouched.
        A retry within the same generation cannot supply this boundary receipt.
        """
        nonlocal _discovered
        ownership = _application.component('ownership')
        if not ownership.is_current(tokens) or any(
                arena.state.phase != 2 for arena in ownership.snapshots()):
            raise RuntimeError('Repair retirement requires the current native replacement')
        _pending.clear()
        _discovered = ownership.generation_key()


    def native_available():
        try: import bpy
        except ImportError: return None
        return hasattr(bpy.data, 'node_groups')


    def stack_values(values):
        return tuple((id_key(grade), enabled, direction, key, tuple(focus))
                     for grade, enabled, direction, key, focus in values)


    def restore_stack_values(values):
        return [(None if identity is None else require_tree(identity), enabled, direction, key, tuple(focus))
                for identity, enabled, direction, key, focus in values]


    def kernel_snapshot(node):
        return {'node': node_key(node), 'kernel': id_key(node.node_tree),
                'links': tuple((node_key(link.from_node), link.from_socket.name,
                                node_key(link.to_node), link.to_socket.name)
                               for link in node.id_data.links
                               if link.from_node == node or link.to_node == node),
                'values': {socket.name: tuple(socket.default_value)
                           if hasattr(socket.default_value, '__len__')
                           and not isinstance(socket.default_value, str) else socket.default_value
                           for socket in node.inputs if hasattr(socket, 'default_value')}}


    def repair_kernel(record):
        tree = find_tree(record['node']['tree'])
        if tree is None: return  # Native replacement retired the entire owning ID.
        node = find_node(tree, record['node'])
        if node is None: raise RuntimeError('Kernel repair lost its owning node')
        kernel = require_tree(record['kernel'])
        for link in tuple(tree.links):
            if link.from_node == node or link.to_node == node: tree.links.remove(link)
        node.node_tree = kernel
        for name, value in record['values'].items():
            if name in node.inputs: node.inputs[name].default_value = value
        for source, output, target, input_ in record['links']:
            src, dst = find_node(tree, source), find_node(tree, target)
            if src is None or dst is None: raise RuntimeError('Kernel repair lost a linked node')
            tree.links.new(src.outputs[output], dst.inputs[input_])
        verify_kernel(record)


    def verify_kernel(record, after=False):
        tree = find_tree(record['node']['tree'])
        if tree is None: return
        node = find_node(tree, record['node'])
        if node is None: raise RuntimeError('Kernel receipt lost its owning node')
        expected = record['after_kernel'] if after else record['kernel']
        if node.node_tree != require_tree(expected): raise RuntimeError('Kernel receipt does not match the selected kernel')
        for name, value in record['values'].items():
            if name not in node.inputs: raise RuntimeError('Kernel receipt lost an input socket')
            actual = node.inputs[name].default_value
            if hasattr(actual, '__len__') and not isinstance(actual, str): actual = tuple(actual)
            if actual != value: raise RuntimeError('Kernel receipt does not match input values')
        expected_links = []
        for source, output, target, input_ in record['links']:
            src, dst = find_node(tree, source), find_node(tree, target)
            if src is None or dst is None: raise RuntimeError('Kernel receipt lost a linked node')
            expected_links.append((src.as_pointer(), output, dst.as_pointer(), input_))
        actual_links = [(link.from_node.as_pointer(), link.from_socket.name,
                         link.to_node.as_pointer(), link.to_socket.name) for link in tree.links
                        if link.from_node == node or link.to_node == node]
        if sorted(actual_links) != sorted(expected_links): raise RuntimeError('Kernel receipt does not match its links')


    def repair_driver(record):
        tree = find_tree(record['tree'])
        if tree is None: return
        animation = tree.animation_data
        curve = animation.drivers.find(record['path'], index=record['index']) if animation else None
        if curve is None: raise RuntimeError('Grade repair lost its animation dependency')
        curve.mute = record['mute']
        verify_driver(record)


    def verify_driver(record, after=False):
        tree = find_tree(record['tree'])
        if tree is None: return
        animation = tree.animation_data
        curve = animation.drivers.find(record['path'], index=record['index']) if animation else None
        expected = record['after_mute'] if after else record['mute']
        if curve is None or curve.mute != expected:
            raise RuntimeError('Grade dependency does not match its repair receipt')

    def _set_export_NODE_ID(value):
        nonlocal NODE_ID
        NODE_ID = value

    def _set_export_RECORDS(value):
        nonlocal RECORDS
        RECORDS = value

    def _set_export_SCHEMA(value):
        nonlocal SCHEMA
        SCHEMA = value

    def _set_export__discovered(value):
        nonlocal _discovered
        _discovered = value

    def _set_export__edge_queries(value):
        nonlocal _edge_queries
        _edge_queries = value

    def _set_export__occurrences(value):
        nonlocal _occurrences
        _occurrences = value

    def _set_export__pending(value):
        nonlocal _pending
        _pending = value

    def _set_export__session(value):
        nonlocal _session
        _session = value

    def _set_export_plain(value):
        nonlocal plain
        plain = value

    def _set_export_edge_token(value):
        nonlocal edge_token
        edge_token = value

    def _set_export_edge_count(value):
        nonlocal edge_count
        edge_count = value

    def _set_export_effect_event(value):
        nonlocal effect_event
        effect_event = value

    def _set_export_Transaction(value):
        nonlocal Transaction
        Transaction = value

    def _set_export_pending(value):
        nonlocal pending
        pending = value

    def _set_export_require_clean(value):
        nonlocal require_clean
        require_clean = value

    def _set_export_repair_all(value):
        nonlocal repair_all
        repair_all = value

    def _set_export_id_key(value):
        nonlocal id_key
        id_key = value

    def _set_export_id_kind(value):
        nonlocal id_kind
        id_kind = value

    def _set_export_fresh_identity(value):
        nonlocal fresh_identity
        fresh_identity = value

    def _set_export_trees(value):
        nonlocal trees
        trees = value

    def _set_export_find_tree(value):
        nonlocal find_tree
        find_tree = value

    def _set_export_find_action(value):
        nonlocal find_action
        find_action = value

    def _set_export_find_id(value):
        nonlocal find_id
        find_id = value

    def _set_export_require_tree(value):
        nonlocal require_tree
        require_tree = value

    def _set_export_node_key(value):
        nonlocal node_key
        node_key = value

    def _set_export_find_node(value):
        nonlocal find_node
        find_node = value

    def _set_export_owner_key(value):
        nonlocal owner_key
        owner_key = value

    def _set_export_read_debts(value):
        nonlocal read_debts
        read_debts = value

    def _set_export_write_debt(value):
        nonlocal write_debt
        write_debt = value

    def _set_export_erase_debt(value):
        nonlocal erase_debt
        erase_debt = value

    def _set_export_rebase_record(value):
        nonlocal rebase_record
        rebase_record = value

    def _set_export_discover(value):
        nonlocal discover
        discover = value

    def _set_export_native_available(value):
        nonlocal native_available
        native_available = value

    def _set_export_stack_values(value):
        nonlocal stack_values
        stack_values = value

    def _set_export_restore_stack_values(value):
        nonlocal restore_stack_values
        restore_stack_values = value

    def _set_export_kernel_snapshot(value):
        nonlocal kernel_snapshot
        kernel_snapshot = value

    def _set_export_repair_kernel(value):
        nonlocal repair_kernel
        repair_kernel = value

    def _set_export_verify_kernel(value):
        nonlocal verify_kernel
        verify_kernel = value

    def _set_export_repair_driver(value):
        nonlocal repair_driver
        repair_driver = value

    def _set_export_verify_driver(value):
        nonlocal verify_driver
        verify_driver = value

    return Component(readers={
        "__name__": lambda: __name__,
        'NODE_ID': lambda: NODE_ID,
        'RECORDS': lambda: RECORDS,
        'SCHEMA': lambda: SCHEMA,
        'Transaction': lambda: Transaction,
        '_discovered': lambda: _discovered,
        '_edge_queries': lambda: _edge_queries,
        '_occurrences': lambda: _occurrences,
        '_pending': lambda: _pending,
        '_session': lambda: _session,
        'count': lambda: count,
        'discover': lambda: discover,
        'edge_count': lambda: edge_count,
        'edge_token': lambda: edge_token,
        'effect_event': lambda: effect_event,
        'erase_debt': lambda: erase_debt,
        'find_action': lambda: find_action,
        'find_id': lambda: find_id,
        'find_node': lambda: find_node,
        'find_tree': lambda: find_tree,
        'fresh_identity': lambda: fresh_identity,
        'id_key': lambda: id_key,
        'id_kind': lambda: id_kind,
        'json': lambda: json,
        'kernel_snapshot': lambda: kernel_snapshot,
        'native_available': lambda: native_available,
        'native_replaced': lambda: native_replaced,
        'node_key': lambda: node_key,
        'owner_key': lambda: owner_key,
        'pending': lambda: pending,
        'plain': lambda: plain,
        'read_debts': lambda: read_debts,
        'rebase_record': lambda: rebase_record,
        'repair_all': lambda: repair_all,
        'repair_driver': lambda: repair_driver,
        'repair_kernel': lambda: repair_kernel,
        'require_clean': lambda: require_clean,
        'require_tree': lambda: require_tree,
        'restore_stack_values': lambda: restore_stack_values,
        'stack_values': lambda: stack_values,
        'text': lambda: text,
        'trees': lambda: trees,
        'uuid4': lambda: uuid4,
        'verify_driver': lambda: verify_driver,
        'verify_kernel': lambda: verify_kernel,
        'write_debt': lambda: write_debt,
    }, writers={
        'NODE_ID': _set_export_NODE_ID,
        'RECORDS': _set_export_RECORDS,
        'SCHEMA': _set_export_SCHEMA,
        '_discovered': _set_export__discovered,
        '_edge_queries': _set_export__edge_queries,
        '_occurrences': _set_export__occurrences,
        '_pending': _set_export__pending,
        '_session': _set_export__session,
    'plain': _set_export_plain, 'edge_token': _set_export_edge_token, 'edge_count': _set_export_edge_count, 'effect_event': _set_export_effect_event, 'Transaction': _set_export_Transaction, 'pending': _set_export_pending, 'require_clean': _set_export_require_clean, 'repair_all': _set_export_repair_all, 'id_key': _set_export_id_key, 'id_kind': _set_export_id_kind, 'fresh_identity': _set_export_fresh_identity, 'trees': _set_export_trees, 'find_tree': _set_export_find_tree, 'find_action': _set_export_find_action, 'find_id': _set_export_find_id, 'require_tree': _set_export_require_tree, 'node_key': _set_export_node_key, 'find_node': _set_export_find_node, 'owner_key': _set_export_owner_key, 'read_debts': _set_export_read_debts, 'write_debt': _set_export_write_debt, 'erase_debt': _set_export_erase_debt, 'rebase_record': _set_export_rebase_record, 'discover': _set_export_discover, 'native_available': _set_export_native_available, 'stack_values': _set_export_stack_values, 'restore_stack_values': _set_export_restore_stack_values, 'kernel_snapshot': _set_export_kernel_snapshot, 'repair_kernel': _set_export_repair_kernel, 'verify_kernel': _set_export_verify_kernel, 'repair_driver': _set_export_repair_driver, 'verify_driver': _set_export_verify_driver})
