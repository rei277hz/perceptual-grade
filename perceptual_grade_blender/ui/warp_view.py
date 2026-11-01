"""Transient, editor-local Warp view bounds held for one complete gesture."""

from ..integration.components import Component

def create(_application):
    from dataclasses import dataclass
    import math
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve


    @dataclass(eq=False)
    class Lock:
        editor: tuple
        node_ref: object
        pointer: int
        tone: int
        radius: float

        @property
        def node(self):return resolve(self.node_ref)


    _locks = {}


    def editor_key(context):
        """A separate canvas in another editor must never inherit a drag's fit."""
        try:
            return tuple(getattr(context, name).as_pointer()
                         for name in ('window', 'area', 'region'))
        except (AttributeError, ReferenceError, RuntimeError):
            return None


    def begin(editor, node, tone, radius):
        """Capture before the first write; the returned token owns its cleanup."""
        if editor is None or not math.isfinite(radius) or radius <= 0:
            return None
        token = Lock(editor, retain(node,'warp.view'), node.as_pointer(), int(tone), float(radius))
        _locks[editor] = token
        return token


    def radius(editor, node, tone):
        """Return a held radius only for the same editor, target and tone."""
        token = _locks.get(editor)
        if token is None:
            return None
        try:
            owner = token.node
            matches = (owner is not None and owner.as_pointer() == node.as_pointer() == token.pointer
                       and token.tone == int(tone))
        except (ReferenceError, RuntimeError):
            matches = False
        if matches:
            return token.radius
        end(token)
        return None


    def end(token):
        if token is not None and _locks.get(token.editor) is token:
            _locks.pop(token.editor, None)


    def prune(live_editors, valid):
        for token in tuple(_locks.values()):
            try:
                owner = token.node
                keep = owner is not None and token.editor in live_editors and valid(owner)
            except (ReferenceError, RuntimeError):
                keep = False
            if not keep:
                end(token)


    def clear():
        _locks.clear()

    def _set_export__locks(value):
        nonlocal _locks
        _locks = value

    def _set_export_Lock(value):
        nonlocal Lock
        Lock = value

    def _set_export_editor_key(value):
        nonlocal editor_key
        editor_key = value

    def _set_export_begin(value):
        nonlocal begin
        begin = value

    def _set_export_radius(value):
        nonlocal radius
        radius = value

    def _set_export_end(value):
        nonlocal end
        end = value

    def _set_export_prune(value):
        nonlocal prune
        prune = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Lock': lambda: Lock,
        '_locks': lambda: _locks,
        'begin': lambda: begin,
        'clear': lambda: clear,
        'dataclass': lambda: dataclass,
        'editor_key': lambda: editor_key,
        'end': lambda: end,
        'math': lambda: math,
        'prune': lambda: prune,
        'radius': lambda: radius,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
    }, writers={
        '_locks': _set_export__locks,
    'Lock': _set_export_Lock, 'editor_key': _set_export_editor_key, 'begin': _set_export_begin, 'radius': _set_export_radius, 'end': _set_export_end, 'prune': _set_export_prune, 'clear': _set_export_clear})
