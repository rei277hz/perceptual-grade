"""Standalone GNU gettext catalogs with semantic, locale-independent IDs.

JSON sources are compiled with Babel at build time. Runtime uses only the
Python standard library. Native RNA strings are resolved through en.json at
registration because Blender stores display strings rather than message IDs.
"""

from ..integration.components import Component

def create(_application):
    import unicodedata
    from ...perceptual_grade_core.localization import catalogs
    from ...perceptual_grade_core.localization.catalogs import CATALOGS, CONTEXT, Message, M

    _ALIASES = {'zh_CN': 'zh_HANS'}
    _registered = False


    def _language(preference):
        try:
            import bpy
        except ModuleNotFoundError:
            return 'en'
        view = bpy.context.preferences.view
        if not getattr(view, preference):
            return 'en'
        locale = view.language
        if locale == 'DEFAULT':
            locale = bpy.app.translations.locale
        locale = _ALIASES.get(locale, locale)
        if locale in CATALOGS:
            return locale
        base = locale.split('_', 1)[0]
        return base if base in CATALOGS else 'en'


    def language():
        return _language('use_translate_interface')


    def tooltip_language():
        return _language('use_translate_tooltips')


    def tr(message_id, *, locale=None, **values):
        return catalogs.tr(message_id, locale=locale or language(), **values)


    def native(message_id, **values):
        """Resolve the host's RNA source string from the English catalog."""
        return tr(message_id, locale='en', **values)


    def tip(message_id, **values):
        """Dynamic operator/socket tooltip, following the tooltip preference."""
        return tr(message_id, locale=tooltip_language(), **values)


    def display(value):
        return tr(value) if isinstance(value, Message) else str(value)


    def error_text(value, *, locale=None):
        locale = locale or language()
        return catalogs.issue_text(value, locale=locale, unknown=tr('outcome.failed', locale=locale))


    def wrap(message, columns):
        """Wrap the resolved text, counting CJK glyphs as two cells."""
        message = display(message)
        rows = []; row = ''; width = 0
        for token in message.split(' '):
            pieces = list(token) if any(unicodedata.east_asian_width(c) in 'WF' for c in token) else [token]
            for i, piece in enumerate(pieces):
                gap = ' ' if row and i == 0 and len(pieces) == 1 else ''
                size = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in gap + piece)
                if row and width + size > columns:
                    rows.append(row); row = ''; width = 0; size -= len(gap); gap = ''
                row += gap + piece; width += size
        if row:
            rows.append(row)
        return rows


    def host_messages(locale):
        """Blender bridge: custom-context names, default-context tooltips.

        Blender performs tooltip lookup using its default context, independently
        of a property's translation_context. General short labels stay scoped.
        """
        messages = {}
        for message_id in CATALOGS['en']:
            source = native(message_id)
            translated = tr(message_id, locale=locale)
            messages[(CONTEXT, source)] = translated
            if message_id.startswith('tooltip.') or message_id in {
                'product.name', 'socket.reference', 'socket.color', 'socket.alpha', 'socket.image',
                'bake.output.label', 'bake.vector.tooltip'
            }:
                messages[('*', source)] = translated
        return messages


    def register():
        nonlocal _registered
        import bpy
        unregister()
        messages = {locale: host_messages(locale) for locale in CATALOGS if locale != 'en'}
        messages.update({alias: messages[locale] for alias, locale in _ALIASES.items() if locale in messages})
        bpy.app.translations.register(__name__, messages)
        _registered = True


    def unregister():
        nonlocal _registered
        import bpy
        if _registered:
            bpy.app.translations.unregister(__name__)
            _registered = False


    def update_node_title(node):
        """Follow language changes while keeping a user's custom node label."""
        stacks = _application.component('graph.stacks')
        label = tr('node.stack.title' if stacks.is_stack(node) else 'node.grade.title')
        if node.label == node.get('_pg_auto_label') and node.label != label:
            node.label = label
            node['_pg_auto_label'] = label


    def marker(kind, value):
        return tr(f'marker.{kind}.{value}')

    def _set_export__ALIASES(value):
        nonlocal _ALIASES
        _ALIASES = value

    def _set_export__registered(value):
        nonlocal _registered
        _registered = value

    def _set_export__language(value):
        nonlocal _language
        _language = value

    def _set_export_language(value):
        nonlocal language
        language = value

    def _set_export_tooltip_language(value):
        nonlocal tooltip_language
        tooltip_language = value

    def _set_export_tr(value):
        nonlocal tr
        tr = value

    def _set_export_native(value):
        nonlocal native
        native = value

    def _set_export_tip(value):
        nonlocal tip
        tip = value

    def _set_export_display(value):
        nonlocal display
        display = value

    def _set_export_error_text(value):
        nonlocal error_text
        error_text = value

    def _set_export_wrap(value):
        nonlocal wrap
        wrap = value

    def _set_export_host_messages(value):
        nonlocal host_messages
        host_messages = value

    def _set_export_register(value):
        nonlocal register
        register = value

    def _set_export_unregister(value):
        nonlocal unregister
        unregister = value

    def _set_export_update_node_title(value):
        nonlocal update_node_title
        update_node_title = value

    def _set_export_marker(value):
        nonlocal marker
        marker = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CATALOGS': lambda: CATALOGS,
        'CONTEXT': lambda: CONTEXT,
        'M': lambda: M,
        'Message': lambda: Message,
        '_ALIASES': lambda: _ALIASES,
        '_language': lambda: _language,
        '_registered': lambda: _registered,
        'catalogs': lambda: catalogs,
        'display': lambda: display,
        'error_text': lambda: error_text,
        'host_messages': lambda: host_messages,
        'language': lambda: language,
        'marker': lambda: marker,
        'native': lambda: native,
        'register': lambda: register,
        'tip': lambda: tip,
        'tooltip_language': lambda: tooltip_language,
        'tr': lambda: tr,
        'unicodedata': lambda: unicodedata,
        'unregister': lambda: unregister,
        'update_node_title': lambda: update_node_title,
        'wrap': lambda: wrap,
    }, writers={
        '_ALIASES': _set_export__ALIASES,
        '_registered': _set_export__registered,
    '_language': _set_export__language, 'language': _set_export_language, 'tooltip_language': _set_export_tooltip_language, 'tr': _set_export_tr, 'native': _set_export_native, 'tip': _set_export_tip, 'display': _set_export_display, 'error_text': _set_export_error_text, 'wrap': _set_export_wrap, 'host_messages': _set_export_host_messages, 'register': _set_export_register, 'unregister': _set_export_unregister, 'update_node_title': _set_export_update_node_title, 'marker': _set_export_marker})
