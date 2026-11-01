"""Scope-only comparison signal, aligned with the primary image's sample grid.

The comparison never enters grading, driver histograms or robust stats.
Warp may display it as a second sample cloud on its appearance pad.
Stage 3 reuses primary Stage-2 tone memberships at the same UV positions.
"""

from ..integration.components import Component

def create(_application):


    from ...perceptual_grade_core.model.scope_reference import evaluate, select

    SOCKET='Color Reference'
    VISIBILITY='_pg_comparison_visible'
    _CONNECTIONS={}
    _REVEAL=set()


    def connected(node):
        socket=node.inputs.get(SOCKET)
        return socket is not None and socket.is_linked


    def sync_connection(node,persist=True):
        """Reveal newly connected/replaced signals; drawing defers persistent writes."""
        key=node.as_pointer();socket=node.inputs.get(SOCKET)
        mark=tuple((link.from_node.as_pointer(),link.from_socket.identifier)
                   for link in socket.links) if socket is not None and socket.is_linked else ()
        previous=_CONNECTIONS.get(key)
        if key in _CONNECTIONS and mark and mark!=previous:_REVEAL.add(key)
        _CONNECTIONS[key]=mark
        if not mark:_REVEAL.discard(key)
        if persist and key in _REVEAL:
            node[VISIBILITY]=True;_REVEAL.discard(key)
        return bool(mark)


    def visible(node):
        linked=sync_connection(node,persist=False)
        return linked and (node.as_pointer() in _REVEAL or node.get(VISIBILITY,True))


    def toggle(node):
        sync_connection(node)
        node[VISIBILITY]=not node.get(VISIBILITY,True)


    def get(node,base):
        return base.get('_comparison') if base is not None and visible(node) else None


    def selected(node,base):
        return base.get('_comparison_selected') if base is not None and visible(node) else None






    def clear():
        _CONNECTIONS.clear();_REVEAL.clear()

    def _set_export_SOCKET(value):
        nonlocal SOCKET
        SOCKET = value

    def _set_export_VISIBILITY(value):
        nonlocal VISIBILITY
        VISIBILITY = value

    def _set_export__CONNECTIONS(value):
        nonlocal _CONNECTIONS
        _CONNECTIONS = value

    def _set_export__REVEAL(value):
        nonlocal _REVEAL
        _REVEAL = value

    def _set_export_connected(value):
        nonlocal connected
        connected = value

    def _set_export_sync_connection(value):
        nonlocal sync_connection
        sync_connection = value

    def _set_export_visible(value):
        nonlocal visible
        visible = value

    def _set_export_toggle(value):
        nonlocal toggle
        toggle = value

    def _set_export_get(value):
        nonlocal get
        get = value

    def _set_export_selected(value):
        nonlocal selected
        selected = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'SOCKET': lambda: SOCKET,
        'VISIBILITY': lambda: VISIBILITY,
        '_CONNECTIONS': lambda: _CONNECTIONS,
        '_REVEAL': lambda: _REVEAL,
        'clear': lambda: clear,
        'connected': lambda: connected,
        'evaluate': lambda: evaluate,
        'get': lambda: get,
        'select': lambda: select,
        'selected': lambda: selected,
        'sync_connection': lambda: sync_connection,
        'toggle': lambda: toggle,
        'visible': lambda: visible,
    }, writers={
        'SOCKET': _set_export_SOCKET,
        'VISIBILITY': _set_export_VISIBILITY,
        '_CONNECTIONS': _set_export__CONNECTIONS,
        '_REVEAL': _set_export__REVEAL,
    'connected': _set_export_connected, 'sync_connection': _set_export_sync_connection, 'visible': _set_export_visible, 'toggle': _set_export_toggle, 'get': _set_export_get, 'selected': _set_export_selected, 'clear': _set_export_clear})
