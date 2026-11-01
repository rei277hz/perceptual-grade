"""Retain independently published scope components without copying their arrays.

One scheduler owner retains a finite set of semantic slots: Stage 1, Stage 2,
the Stack overview, and four Warp/four J/C views. Worker admission remains in
the scheduler; this adapter applies the component policy to its receipts.
"""

from ..integration.components import Component

def create(_application):
    from types import MappingProxyType
    from ...perceptual_grade_core.graph import scheduler_policy as policy


    def descriptors(spec, kind, intern):
        analysis = spec.get('_warp_analysis_key', spec['key'])
        demands = spec.get('_scope_demands')
        if demands is None:
            name = 'linear' if kind == 'linear' else 'tone' if spec.get('stage') == 2 else 'overview'
            return ((name, intern(('scope-slot', name)), intern(analysis), 0),)
        return tuple((f'{tone}:{"warp" if warp else "jc"}',
            intern(('scope-slot', tone, warp)), intern((analysis, tone, warp)),
            intern(tuple(view['key'] for view in spec.get('_warp_views', ())
                         if view.get('tone', tone) == tone)) if warp else 0)
            for tone, warp in demands)


    def request(record, ticket):
        slots = record.setdefault('scope_slots', {})
        names = tuple(row[0] for row in ticket.display)
        changed = names != record.get('scope_visible')
        record['scope_visible'] = names
        for name, slot, analysis, geometry in ticket.display:
            entry = slots.get(name)
            if entry is None:
                store = _application.app.store(machine=_application.machines['display'])
                store.dispatch(policy.ScopeDisplayEvent('invalidate', source=ticket.display_source, slot=slot))
                entry = slots[name] = dict(store=store, visual=None, statistics=None)
            before = entry['store'].state
            entry['store'].dispatch(policy.ScopeDisplayEvent(
                'request', source=ticket.display_source, slot=slot, analysis=analysis,
                geometry=geometry, epoch=ticket.token))
            changed |= (before.desired_analysis, before.desired_geometry) != (analysis, geometry)
        if changed:
            record.pop('scope_composed', None)


    def publish(record, ticket, payload, intern):
        slots = record['scope_slots']
        active = payload.get('_active_colors')
        stats_only = payload.get('_scope_part') == 'stats'
        capture = payload.get('_scope_request', ticket.token)
        changed = False
        for name, slot, analysis, geometry in ticket.display:
            entry = slots.get(name)
            if entry is None:
                continue
            value = active.get(name) if active is not None else payload
            if value is None:
                continue
            event = dict(source=ticket.display_source, slot=slot, analysis=analysis,
                         geometry=geometry, epoch=ticket.token, capture=intern(capture), validated=True)
            # Missing descriptors mean no new raster. A present all-zero raster is
            # a valid empty cloud and must replace its predecessor.
            visual_present = not stats_only and (not name.endswith(':warp') or bool(value.get('_warp_clouds')))
            if visual_present:
                effect = entry['store'].dispatch(policy.ScopeDisplayEvent('visual', **event))
                if effect.replace_visual:
                    entry.update(visual=value, envelope=payload, visual_request=capture)
                    changed = True
            if active is not None and name.endswith(':warp') and value.get('neutral') is not None:
                effect = entry['store'].dispatch(policy.ScopeDisplayEvent('statistics', **event))
                state = entry['store'].state
                if effect.replace_statistics:
                    entry.update(statistics=value['neutral'], statistics_request=capture, statistics_view=value)
                    if ticket.token >= state.visual_epoch:
                        entry['envelope'] = payload
                    changed = True
        if changed:
            record.pop('scope_composed', None)


    def result(record):
        if 'scope_composed' in record:
            return record['scope_composed']
        slots = record.get('scope_slots', {})
        visible = record.get('scope_visible', ())
        if not visible:
            return None
        if ':' not in visible[0]:
            entry = slots.get(visible[0], {})
            return entry.get('visual')
        colors = {}
        envelope = None
        newest = -1
        for name in visible:
            entry = slots.get(name)
            if entry is None or (entry['visual'] is None and entry['statistics'] is None):
                continue
            state = entry['store'].state
            item = dict(entry['visual'] if entry['visual'] is not None else entry['statistics_view'],
                _visual_request=entry.get('visual_request'),
                _visual_analysis=state.visual_analysis, _visual_epoch=state.visual_epoch,
                _display_current=bool(policy.scope_display_current(state, name.endswith(':warp'))))
            if name.endswith(':warp'):
                item.update(neutral=entry['statistics'],
                    _statistics_request=entry.get('statistics_request'),
                    _statistics_analysis=state.stats_analysis, _statistics_epoch=state.stats_epoch,
                    statistics_status='ready' if state.stats_available and
                        state.stats_analysis == state.desired_analysis else 'pending')
            colors[name] = MappingProxyType(item)
            if max(state.visual_epoch, state.stats_epoch) > newest:
                envelope = entry['envelope']
                newest = max(state.visual_epoch, state.stats_epoch)
        if not colors:
            return None
        value = MappingProxyType(dict(envelope, _active_colors=MappingProxyType(colors),
            _warp_clouds=tuple(cloud for item in colors.values() for cloud in item.get('_warp_clouds', ())),
            _active_statistics_pending=tuple(int(name.split(':')[0]) for name, item in colors.items()
                if name.endswith(':warp') and item['statistics_status'] != 'ready')))
        record['scope_composed'] = value
        return value


    def current(record):
        slots = record.get('scope_slots', {})
        visible = record.get('scope_visible', ())
        return bool(visible) and all(name in slots and policy.scope_display_current(
            slots[name]['store'].state, name.endswith(':warp')) for name in visible)

    def _set_export_descriptors(value):
        nonlocal descriptors
        descriptors = value

    def _set_export_request(value):
        nonlocal request
        request = value

    def _set_export_publish(value):
        nonlocal publish
        publish = value

    def _set_export_result(value):
        nonlocal result
        result = value

    def _set_export_current(value):
        nonlocal current
        current = value

    return Component(readers={
        "__name__": lambda: __name__,
        'MappingProxyType': lambda: MappingProxyType,
        'current': lambda: current,
        'descriptors': lambda: descriptors,
        'policy': lambda: policy,
        'publish': lambda: publish,
        'request': lambda: request,
        'result': lambda: result,
    }, writers={
    'descriptors': _set_export_descriptors, 'request': _set_export_request, 'publish': _set_export_publish, 'result': _set_export_result, 'current': _set_export_current})
