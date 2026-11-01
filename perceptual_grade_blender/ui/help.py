"""Localized hover descriptions for the drawn grading panel.

Control descriptions come from the same bindings as the numeric sidebar.
Picking uses control geometry only. The existing panel timer advances the
hover delay; neither hovering nor tooltip drawing evaluates an image.
"""

from ..integration.components import Component

def create(_application):
    from dataclasses import dataclass
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    import math
    import time
    i18n = _application.component('localization.i18n')
    from ...perceptual_grade_core.localization.controls import control_label, control_tooltip
    in_rect = _application.component('ui.geometry').in_rect; hit_handle = _application.component('ui.geometry').hit_handle

    DELAY = .55
    MOTION_TOLERANCE = 3.


    @dataclass(frozen=True)
    class Section:
        description: str
        label: str = ''
        values: tuple = ()

        def text(self):
            locale = i18n.tooltip_language()
            values = {key: i18n.tr(value, locale=locale) if isinstance(value, i18n.Message) else value
                      for key, value in self.values}
            return i18n.tip(self.description, **values)


    @dataclass(frozen=True)
    class Help:
        title: str
        sections: tuple


    def explanation(title, description, **values):
        return Help(title, (Section(description, values=tuple(values.items())),))


    def controls(stage, fields, title=None, gesture=None):
        def label(field):
            return control_label('remap_fit' if stage=='tone' and field=='fit' else field)
        sections = tuple(Section(control_tooltip(stage, field), label(field) if len(fields)>1 else '')
                         for field in fields)
        if gesture:
            sections += (Section(gesture),)
        return Help(title or label(fields[0]), sections)


    def tone_name(index, black=False):
        names = ('black', 'shadows', 'midtones', 'white', 'highlights') if black else ('shadows', 'midtones', 'white', 'highlights')
        return i18n.M('tone.'+names[index])


    def at(context, node, box, point):
        """Use the same handle precedence as the click router, including overlaps."""
        nodes = _application.component('graph.nodes')
        diagram = _application.component('ui.diagram')
        from ...perceptual_grade_core.model import linear as linear
        editing = _application.component('graph.editing')
        stack=editing.is_target(node)
        for key, stage in (('linear_tab', 'linear'), ('tone_tab', 'tone'), ('color_tab', 'color')):
            if in_rect(point, box[key]):
                return explanation('stage.'+stage+'.label', 'tooltip.stage.'+stage)
        if in_rect(point, box['close']):
            return explanation('action.hide_panel.label', 'tooltip.action.hide_panel')
        if in_rect(point, box['header']):
            return explanation('product.name', 'tooltip.gizmo.move_panel')
        if not nodes.editable(node):
            return None
        stage = getattr(context.window_manager, 'pg_stage', 'LINEAR')
        space = nodes.grading_space(node)
        if stage == 'LINEAR' and in_rect(point, box['vectorscope']):
            if stack:
                return explanation('scope.stack.output','tooltip.scope.stack.output.drag')
            help = controls('linear', ('source_x', 'source_y'), 'scope.vectorscope.draggable', 'tooltip.gizmo.scope_white')
            return with_reference(help, node)
        if diagram.comparison_hit(context, node, box, point):
            return explanation('socket.reference', 'tooltip.gizmo.reference_toggle')
        if stage == 'CHROMATIC':
            color = _application.component('ui.color'); geometry = _application.component('ui.color_geometry')
            action = color.hit(node, space, box, point)
            kind = action['kind']
            if kind=='warp_dot':
                return explanation('family.warp.label','tooltip.warp.target')
            if kind in {'slider', 'dot'}:
                gesture = None
                if kind=='dot':
                    if node.pg_chromatic.family=='WARP':gesture='tooltip.gizmo.selected_dot'
                    elif not action.get('on_handle'):gesture='tooltip.gizmo.edit_selected_dot'
                return controls('color', action['fields'], gesture=gesture)
            if kind == 'family':
                key = {'J':'j', 'C':'c', 'WARP':'warp'}[action['value']]
                return explanation('family.'+key+'.label', 'tooltip.family.'+key)
            if kind == 'driver':
                if action.get('source') == 'indicator':
                    name = tone_name(action['tone'])
                    return explanation(str(name), 'tooltip.gizmo.chroma_indicator', tone=name)
                field = node.pg_chromatic.family.lower()+'_'+action['value']+'_'+str(node.pg_chromatic.active_tone)
                return controls('color', (field,), gesture='tooltip.gizmo.driver_selector')
            if kind == 'tone':
                name = tone_name(action['value'])
                message = ('tooltip.gizmo.chroma_indicator' if action.get('source')=='indicator'
                           and node.pg_chromatic.family in {'J','C'} else 'tooltip.gizmo.select_tone')
                return explanation(str(name), message, tone=name)
            if node.pg_chromatic.family!='WARP' and in_rect(point, geometry.square(box['vectorscope'])):
                return with_reference(explanation('scope.vectorscope', 'tooltip.gizmo.vectorscope'), node)
            if node.pg_chromatic.family!='WARP' and in_rect(point, box['waveform']):
                return with_reference(explanation('scope.waveform', 'tooltip.gizmo.color_waveform'), node)
            if node.pg_chromatic.family == 'WARP':
                x,y=box['warp_statistics'];s=box['scale']
                width=min(box['ab'][2],max(80.,len(i18n.tr('warp.statistics'))*7.)*s)
                if in_rect(point, (x-4*s,y-4*s,width+8*s,15*s)):
                    return explanation('diagram.adaptance.statistics', 'tooltip.gizmo.neutral_statistics')
        elif stage == 'REMAP':
            tone = _application.component('ui.tone')
            from ...perceptual_grade_core.model import tone as tonal_math
            if in_rect(point, box['exposure']):
                return controls('linear', ('exposure',))
            if in_rect(point, box['fit']):
                return controls('tone', ('fit',))
            if in_rect(point, box['peak']):
                return explanation('control.peak_nits.label', 'tooltip.gizmo.peak_anchor')
            model = tonal_math.model(nodes.read_tone(node), space)
            index = None if stack else tone.hit_level(point, box, model)
            if index is not None:
                return controls('tone', (tonal_math.LEVEL_FIELDS[index],), gesture='tooltip.gizmo.input_level')
            index = None if stack else tone.hit_level(point, box, model, 'after')
            if index is not None:
                name = tone_name(index, black=True)
                return explanation(str(name), 'tooltip.gizmo.target_level', tone=name)
            index = tone.hit_curve(point, box, model, context.window_manager.pg_selected_tone)
            if index is not None:
                return controls('tone', (tonal_math.LEVEL_FIELDS[index],), gesture='tooltip.gizmo.input_curve')
            for side, title, description in (('before','scope.before','tooltip.gizmo.before_waveform'),
                                             ('after','scope.after','tooltip.gizmo.after_waveform')):
                if in_rect(point, box[side]):
                    if stack:
                        suffix='input' if side=='before' else 'output'
                        return explanation('scope.stack.'+suffix,'tooltip.scope.stack.'+suffix)
                    return explanation(title, description)
            if in_rect(point, box['chart'].rect):
                return explanation('diagram.tone_curve', 'tooltip.gizmo.tone_curve')
        else:
            if in_rect(point, box['exposure']):
                return controls('linear', ('exposure',))
            if in_rect(point, box['balance']):
                return controls('linear', ('cr_balance','yb_balance'), 'gizmo.balance.label', 'tooltip.gizmo.balance')
            geometry = linear.diagram_geometry(nodes.read_parameters(node), space)
            selected = hit_handle(point, box['chart'], geometry, 7*box['scale'])
            if selected:
                axis,_ = selected
                if axis == 'white':
                    return controls('linear', ('source_x','source_y'), 'white.source.label', 'tooltip.gizmo.white_xy')
                title = {'cr':'gizmo.cr.label', 'yb':'gizmo.yb.label'}[axis]
                return controls('linear', (axis+'_spread',axis+'_heading'), title, 'tooltip.gizmo.direction')
            if math.dist(point, box['chart'].screen(geometry['destination'])) <= 7*box['scale']:
                return explanation('white.destination.label', 'tooltip.gizmo.working_white')
        return None


    def with_reference(help, node):
        comparison = _application.component('scopes.reference')
        if comparison.connected(node):
            return Help(help.title, help.sections+(Section('tooltip.gizmo.reference_toggle'),))
        return help


    def enabled(context):
        view = getattr(context.preferences, 'view', None)
        return getattr(view, 'show_tooltips', True)


    def identity(context, node):
        ui = node.pg_chromatic
        return (context.area.as_pointer(), context.region.as_pointer(), node.as_pointer(),
                context.window_manager.pg_stage, ui.family, ui.active_tone, ui.active_chroma, ui.active_handle,
                getattr(context.window_manager, 'pg_selected_tone', 3))


    @dataclass
    class Pending:
        owner: tuple
        help: Help
        point: tuple
        anchor: tuple
        scale: float
        started: float
        region_ref: object
        ready: bool = False

        @property
        def region(self):return resolve(self.region_ref)


    _states = {}


    def clear(window=None):
        keys = tuple(_states) if window is None else (window.as_pointer(),)
        for key in keys:
            state = _states.pop(key, None)
            if state and state.ready:
                try:
                    region=state.region
                    if region is not None:region.tag_redraw()
                except (ReferenceError, RuntimeError):pass


    def update(window, region, owner, point, scale, help, now=None):
        if window is None:
            return
        if help is None:
            clear(window);return
        now = time.monotonic() if now is None else now
        key = window.as_pointer();old = _states.get(key)
        if old and old.owner==owner and old.help==help and math.dist(point,old.anchor)<=MOTION_TOLERANCE*scale:
            old.point=point
            return
        clear(window)
        _states[key] = Pending(owner, help, point, point, scale, now, retain(region,'help.region'))


    def poll(now=None):
        now = time.monotonic() if now is None else now
        for key,state in tuple(_states.items()):
            region=state.region
            if region is None:
                _states.pop(key,None)
                continue
            if not state.ready and now-state.started>=DELAY:
                state.ready=True
                try:region.tag_redraw()
                except (ReferenceError, RuntimeError):_states.pop(key, None)


    def current(context, node):
        if context.window is None or not enabled(context):
            return None
        state = _states.get(context.window.as_pointer())
        return state if state and state.ready and state.owner==identity(context,node) else None


    def prune(live):
        for key in set(_states)-live:
            _states.pop(key, None)

    def _set_export_DELAY(value):
        nonlocal DELAY
        DELAY = value

    def _set_export_MOTION_TOLERANCE(value):
        nonlocal MOTION_TOLERANCE
        MOTION_TOLERANCE = value

    def _set_export__states(value):
        nonlocal _states
        _states = value

    def _set_export_Section(value):
        nonlocal Section
        Section = value

    def _set_export_Help(value):
        nonlocal Help
        Help = value

    def _set_export_explanation(value):
        nonlocal explanation
        explanation = value

    def _set_export_controls(value):
        nonlocal controls
        controls = value

    def _set_export_tone_name(value):
        nonlocal tone_name
        tone_name = value

    def _set_export_at(value):
        nonlocal at
        at = value

    def _set_export_with_reference(value):
        nonlocal with_reference
        with_reference = value

    def _set_export_enabled(value):
        nonlocal enabled
        enabled = value

    def _set_export_identity(value):
        nonlocal identity
        identity = value

    def _set_export_Pending(value):
        nonlocal Pending
        Pending = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_update(value):
        nonlocal update
        update = value

    def _set_export_poll(value):
        nonlocal poll
        poll = value

    def _set_export_current(value):
        nonlocal current
        current = value

    def _set_export_prune(value):
        nonlocal prune
        prune = value

    return Component(readers={
        "__name__": lambda: __name__,
        'DELAY': lambda: DELAY,
        'Help': lambda: Help,
        'MOTION_TOLERANCE': lambda: MOTION_TOLERANCE,
        'Pending': lambda: Pending,
        'Section': lambda: Section,
        '_states': lambda: _states,
        'at': lambda: at,
        'clear': lambda: clear,
        'control_label': lambda: control_label,
        'control_tooltip': lambda: control_tooltip,
        'controls': lambda: controls,
        'current': lambda: current,
        'dataclass': lambda: dataclass,
        'enabled': lambda: enabled,
        'explanation': lambda: explanation,
        'hit_handle': lambda: hit_handle,
        'i18n': lambda: i18n,
        'identity': lambda: identity,
        'in_rect': lambda: in_rect,
        'math': lambda: math,
        'poll': lambda: poll,
        'prune': lambda: prune,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'time': lambda: time,
        'tone_name': lambda: tone_name,
        'update': lambda: update,
        'with_reference': lambda: with_reference,
    }, writers={
        'DELAY': _set_export_DELAY,
        'MOTION_TOLERANCE': _set_export_MOTION_TOLERANCE,
        '_states': _set_export__states,
    'Section': _set_export_Section, 'Help': _set_export_Help, 'explanation': _set_export_explanation, 'controls': _set_export_controls, 'tone_name': _set_export_tone_name, 'at': _set_export_at, 'with_reference': _set_export_with_reference, 'enabled': _set_export_enabled, 'identity': _set_export_identity, 'Pending': _set_export_Pending, 'clear': _set_export_clear, 'update': _set_export_update, 'poll': _set_export_poll, 'current': _set_export_current, 'prune': _set_export_prune})
