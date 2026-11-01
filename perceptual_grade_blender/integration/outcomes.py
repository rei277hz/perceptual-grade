"""Localized UI outcomes; diagnostic exceptions never become native reports.

Only plain values survive a callback. Internal operations still raise normally;
the boundary around a registered Blender callback records their failed outcome.
"""
from dataclasses import dataclass
from functools import wraps
import logging

from .components import Component
from ...blender_runtime.execution.diagnostics import Failure


@dataclass(frozen=True)
class Outcome:
    control: str
    code: str
    values: tuple
    recovery: str | None = None
    diagnostic: str = ''


def create(application):
    i18n = application.component('localization.i18n')
    records = {}
    epoch = None
    log = logging.getLogger(__name__)

    def address(node):
        nonlocal epoch
        # A Session scalar is latched before native replacement. Status drawing
        # and handler failure reporting must never query an actor or acquire RNA.
        current = application.outcome_epoch
        if current != epoch:
            records.clear()
            epoch = current
        if node is None:
            return (0, 0)
        try:
            owner = getattr(node, 'id_data', None)
            return (int(getattr(owner, 'session_uid', 0)), int(node.as_pointer()))
        except (AttributeError, ReferenceError, RuntimeError):
            return (0, 0)

    def target(context):
        if context is None:
            return None
        try:
            for name in ('pg_node', 'pg_stack_node', 'node'):
                node = getattr(context, name, None)
                if node is not None:
                    return node
            space = getattr(context, 'space_data', None)
            tree = getattr(space, 'edit_tree', None)
            return tree.nodes.active if tree is not None else None
        except (AttributeError, ReferenceError, RuntimeError):
            return None

    def set_(node, control, code, values=None, recovery=None):
        def plain(value):
            if value is None or type(value) in (bool, int, float, str):return value
            if isinstance(value, (tuple, list)):return tuple(plain(item) for item in value)
            return str(value)
        value = Outcome(str(control), str(code), tuple(sorted((str(key), plain(item)) for key, item in (values or {}).items())),
                        None if recovery is None else str(recovery))
        records[(address(node), str(control))] = value
        while len(records) > 256:
            records.pop(next(iter(records)))
        return value

    def failure(node, control, error, recovery=None):
        try:
            diagnostic = Failure.capture(error)
        except Exception:
            diagnostic = Failure(type(error).__name__, '', 'Diagnostic formatting was unavailable')
        key = (address(node), str(control))
        previous = records.get(key)
        if previous is None or previous.diagnostic != diagnostic.detail:
            try:
                log.error('Perceptual Grade %s failed:\n%s', control, diagnostic.detail)
            except Exception:
                pass  # A broken logging handler cannot escape a native callback.
        value = Outcome(str(control), 'outcome.failed', (), recovery, diagnostic.detail)
        records[key] = value
        while len(records) > 256:
            records.pop(next(iter(records)))
        return value

    def clear(node=None, control=None):
        key = address(node)
        if node is None and control is None:
            records.clear()
        else:
            for item in tuple(records):
                if item[0] == key and (control is None or item[1] == control):
                    records.pop(item, None)

    def get(node, control):
        return records.get((address(node), str(control)))

    def recovered(recovery):
        """Retire recovery prompts only after that recovery has completed."""
        address(None)
        for key,value in tuple(records.items()):
            if value.recovery == recovery:records.pop(key,None)

    def draw(layout, node=None, control=None):
        key = address(node)
        for (owner, name), value in tuple(records.items()):
            if owner != key or (control is not None and control != name):
                continue
            message = i18n.tr(value.code, **dict(value.values))
            column = layout.column(align=True)
            for line in i18n.wrap(message, 42):
                column.label(text=line, icon='INFO')
            if value.recovery:
                column.operator(value.recovery, text=i18n.tr('outcome.retry'), icon='FILE_REFRESH')

    def callback(fn, *, control=None, fallback=None, node_argument=False, recovery=None, clear_success=True):
        if getattr(fn, '_pg_outcome_boundary', False):
            return fn
        label = control or fn.__name__

        @wraps(fn)
        def guarded(*args, **kwargs):
            node = args[0] if node_argument and args else None
            try:
                is_node = node is not None and hasattr(node, 'inputs') and hasattr(node, 'outputs')
            except (ReferenceError, RuntimeError):
                is_node = False
            if not is_node:node = None
            previous = get(node,label) if clear_success else None
            try:
                result = fn(*args, **kwargs)
                if previous is not None and get(node,label) is previous:
                    clear(node,label)
                return result
            except Exception as error:
                failure(node, label, error, recovery=recovery)
                return fallback() if callable(fallback) else fallback

        guarded._pg_outcome_boundary = True
        return guarded

    def property_(value, *, control='property'):
        options = getattr(value, 'keywords', None)
        if options is None:
            return value
        default = options.get('default')
        if default is None:
            factory = getattr(getattr(value, 'function', None), '__name__', '')
            default = {'FloatProperty': 0., 'IntProperty': 0, 'BoolProperty': False,
                       'StringProperty': '', 'EnumProperty': 0}.get(factory)
            if 'VectorProperty' in factory:
                default = tuple(0. for _ in range(options.get('size', 3)))
        for kind in ('get', 'set', 'update', 'poll', 'search'):
            fn = options.get(kind)
            if fn is not None:
                fallback = default if kind == 'get' else False if kind == 'poll' else () if kind == 'search' else None
                guarded = callback(fn, control=control, fallback=fallback, node_argument=True,
                                   clear_success=kind in ('set','update'))
                # RNA checks co_argcount rather than inspect.signature or the
                # wrapped function. Keep the native callback ABI explicit.
                def exact_arity(guarded, kind):
                    if kind == 'get':
                        def call(owner):return guarded(owner)
                    elif kind == 'search':
                        def call(owner, context, text):return guarded(owner, context, text)
                    else:
                        def call(owner, value):return guarded(owner, value)
                    return wraps(guarded)(call)
                options[kind] = exact_arity(guarded, kind)
        return value

    def protect_class(cls):
        import bpy
        operator = issubclass(cls, bpy.types.Operator)
        for name, value in getattr(cls, '__annotations__', {}).items():
            property_(value, control=name)
        methods = ('execute', 'invoke', 'modal', 'cancel', 'poll', 'description',
                   'draw', 'draw_buttons', 'draw_buttons_ext', 'draw_label')
        for name in methods:
            original = next((base.__dict__[name] for base in cls.__mro__ if name in base.__dict__), None)
            if original is None or getattr(original, '_pg_outcome_boundary', False):
                continue
            class_method = isinstance(original, classmethod)
            fn = original.__func__ if class_method else original
            if not hasattr(fn, '__code__') or getattr(fn, '__module__', '').startswith('bpy'):
                continue
            if getattr(fn, '_pg_outcome_boundary', False):
                continue
            control = str(getattr(cls, 'bl_idname', cls.__name__))

            def boundary(fn=fn, name=name, control=control):
                @wraps(fn)
                def guarded(self, *args, **kwargs):
                    context = args[0] if args else None
                    node = self if name.startswith('draw_') else target(context)
                    previous = get(node,control) if name in ('execute','invoke') else None
                    try:
                        result = fn(self, *args, **kwargs)
                        if (name in ('execute', 'invoke') and result == {'FINISHED'}
                                and previous is not None and get(node,control) is previous):
                            clear(node, control)
                        return result
                    except Exception as error:
                        failure(node, control, error)
                        if name == 'poll':
                            setter = getattr(self, 'poll_message_set', None)
                            if callable(setter):
                                try:setter(i18n.tip('outcome.failed'))
                                except Exception:pass
                            return False
                        if name in ('description', 'draw_label'):
                            return i18n.tip('outcome.failed')
                        if operator and name in ('invoke', 'execute', 'modal'):
                            cleanup = getattr(self, 'cancel', None)
                            if cleanup is not None:
                                try:
                                    cleanup(context)
                                except Exception as cleanup_error:
                                    failure(node, control + '.cleanup', cleanup_error)
                            # Retain Blender's undo snapshot if an unexpected
                            # exception interrupted an undoable structural edit.
                            return {'FINISHED'} if 'UNDO' in getattr(self, 'bl_options', set()) else {'CANCELLED'}
                        return None
                guarded._pg_outcome_boundary = True
                return guarded

            def native_method(guarded, name):
                if name == 'draw_label':
                    def method(self):return guarded(self)
                elif name in ('invoke', 'modal', 'description', 'draw_buttons', 'draw_buttons_ext'):
                    def method(self, context, value):return guarded(self, context, value)
                else:
                    def method(self, context):return guarded(self, context)
                return wraps(guarded)(method)

            wrapped = native_method(boundary(), name)
            setattr(cls, name, classmethod(wrapped) if class_method else wrapped)
        return cls

    return Component({'set': set_, 'failure': failure, 'clear': clear, 'get': get,
                      'draw': draw, 'target': target, 'callback': callback, 'recovered': recovered,
                      'property': property_, 'protect_class': protect_class})
