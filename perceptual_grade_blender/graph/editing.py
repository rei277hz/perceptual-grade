"""Transient per-editor Grade targets, separate from the rendered Stack.

No helper nodes or copied Grade parameters are created for editing. An entry
key identifies an occurrence, including when one Grade occurs more than once.
"""

from ..integration.components import Component

def create(_application):
    grades = _application.component('graph.grades')
    from itertools import count
    OwnedField = _application.component('ownership').OwnedField

    _targets = {}
    _identities = count(1)


    class Target:
        stack = OwnedField("editing.Target.stack")
        pg_grade = OwnedField("editing.Target.pg_grade")

        def __init__(self, stack, entry):
            self.stack = stack
            self.key = entry.key
            self.pg_grade = entry.grade
            # A virtual target is not an RNA object. Its cache identity must survive
            # collection moves and remain removable after its entry is deleted.
            self._identity = -next(_identities)

        @property
        def entry(self):
            entry = next((item for item in self.stack.pg_stack_entries if item.key == self.key), None)
            if entry is None or entry.grade != self.pg_grade:
                raise ReferenceError('The edited Stack entry was removed or replaced')
            return entry

        @property
        def pg_chromatic(self):
            return self.entry.focus

        @property
        def pg_tone(self):
            return self.pg_grade.pg_tone

        @property
        def pg_linear(self):
            return self.pg_grade.pg_linear

        def as_pointer(self):
            return self._identity

        def __getattr__(self, name):
            return getattr(self.stack, name)

        def get(self, name, default=None):
            return self.stack.get(name, default)

        def __getitem__(self, name):
            return self.stack[name]

        def __setitem__(self, name, value):
            self.stack[name] = value


    def is_target(value):
        return isinstance(value, Target)


    def valid(target):
        stacks = _application.component('graph.stacks')
        try:
            return (stacks.is_stack(target.stack) and target.entry.enabled
                    and grades.is_data(target.pg_grade) and grades.local(target.pg_grade)
                    and grades.local(target.stack.id_data))
        except (ReferenceError, RuntimeError):
            return False


    def area_key(context):
        area = getattr(context, 'area', None)
        return area.as_pointer() if area is not None else None


    def shown(context, target):
        """Pencils follow the diagram actually displayed in this editor."""
        return bool(target is not None and
                    getattr(context.window_manager, 'pg_diagram_enabled', False) and
                    grades.core().active_grade(context) == target)


    def active(context, stack):
        key = area_key(context)
        target = _targets.get(key)
        if target is not None and not valid(target):
            _targets.pop(key, None)
            return None
        return target if target is not None and target.stack == stack else None


    def toggle(context, stack, entry):
        """An explicit pencil click is the only way to open a Stack editor."""
        key = area_key(context)
        target = Target(stack, entry)
        if key is None or not valid(target):
            return False
        previous = active(context, stack)
        if previous is not None and previous.key == entry.key and shown(context, previous):
            _targets.pop(key, None)
            return False
        stack.id_data.nodes.active = stack
        stack.select = True
        _targets[key] = target
        context.window_manager.pg_diagram_enabled = True
        return True


    def close(context):
        _targets.pop(area_key(context), None)


    def prune():
        for key, target in tuple(_targets.items()):
            if not valid(target):
                _targets.pop(key, None)


    def rebind(previous, replacement):
        """Follow this session's native/custom representation during save."""
        for target in _targets.values():
            if target.stack == previous:
                target.stack = replacement


    def clear():
        _targets.clear()

    def _set_export__identities(value):
        nonlocal _identities
        _identities = value

    def _set_export__targets(value):
        nonlocal _targets
        _targets = value

    def _set_export_Target(value):
        nonlocal Target
        Target = value

    def _set_export_is_target(value):
        nonlocal is_target
        is_target = value

    def _set_export_valid(value):
        nonlocal valid
        valid = value

    def _set_export_area_key(value):
        nonlocal area_key
        area_key = value

    def _set_export_shown(value):
        nonlocal shown
        shown = value

    def _set_export_active(value):
        nonlocal active
        active = value

    def _set_export_toggle(value):
        nonlocal toggle
        toggle = value

    def _set_export_close(value):
        nonlocal close
        close = value

    def _set_export_prune(value):
        nonlocal prune
        prune = value

    def _set_export_rebind(value):
        nonlocal rebind
        rebind = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'OwnedField': lambda: OwnedField,
        'Target': lambda: Target,
        '_identities': lambda: _identities,
        '_targets': lambda: _targets,
        'active': lambda: active,
        'area_key': lambda: area_key,
        'clear': lambda: clear,
        'close': lambda: close,
        'count': lambda: count,
        'grades': lambda: grades,
        'is_target': lambda: is_target,
        'prune': lambda: prune,
        'rebind': lambda: rebind,
        'shown': lambda: shown,
        'toggle': lambda: toggle,
        'valid': lambda: valid,
    }, writers={
        '_identities': _set_export__identities,
        '_targets': _set_export__targets,
    'Target': _set_export_Target, 'is_target': _set_export_is_target, 'valid': _set_export_valid, 'area_key': _set_export_area_key, 'shown': _set_export_shown, 'active': _set_export_active, 'toggle': _set_export_toggle, 'close': _set_export_close, 'prune': _set_export_prune, 'rebind': _set_export_rebind, 'clear': _set_export_clear})
