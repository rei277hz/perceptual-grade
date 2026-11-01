"""One product enable lifecycle with explicit, lazy component instances."""
from importlib import import_module

from .integration.components import Component, ComponentRef, set_component_metadata
from .integration.registry import COMPONENTS


class Session:
    def __init__(self, machines):
        self.machines = dict(machines)
        self.components = {}
        self._references = {}
        self._constructing = set()
        self.classes = []
        self.handlers = []
        self._handler_boundaries = {}
        self.outcome_epoch = 1
        self.closed = self.closing = False
        self.waiting_for_readers = False
        from .services import create_app
        self.app = create_app(self)

    def component(self, name):
        if name not in COMPONENTS and name != 'integration.lifecycle':
            raise ValueError('Unknown product component: ' + str(name))
        if name not in self._references:
            self._references[name] = ComponentRef(self, name)
        return self._references[name]

    def _component(self, name):
        if name not in self.components:
            if name in self._constructing:
                raise RuntimeError('Cyclic product component initialization: ' + name)
            self._constructing.add(name)
            try:
                module = import_module('.' + name, __package__)
                component = module.create(self)
                if not isinstance(component, Component):
                    raise TypeError('Product factory must return an explicit Component')
                set_component_metadata(component, __file__=module.__file__)
                self.components[name] = component
            finally:
                self._constructing.remove(name)
        return self.components[name]

    def register_class(self, cls):
        import bpy
        self.app.require_open()
        self.component('integration.outcomes').protect_class(cls)
        setattr(cls, '_perceptual_grade_registration', self)
        bpy.utils.register_class(cls)
        if cls not in self.classes:
            self.classes.append(cls)
        self.app.blender.registration.defer(lambda: self.unregister_class(cls))

    def unregister_class(self, cls):
        import bpy
        if getattr(cls, '_perceptual_grade_registration', None) is not self:
            from ..blender_runtime import ForeignOwnerError
            raise ForeignOwnerError('Native class belongs to a different product Session')
        if getattr(cls, 'is_registered', False):
            bpy.utils.unregister_class(cls)

    def add_handler(self, name, callback):
        import bpy
        key = (name, callback)
        if key not in self._handler_boundaries:
            from functools import wraps
            original = callback
            if name in ('load_pre', 'undo_pre', 'redo_pre'):
                @wraps(original)
                def callback(*args):
                    self.outcome_epoch += 1
                    return original(*args)
            self._handler_boundaries[key] = self.component('integration.outcomes').callback(
                callback, control=name, recovery='perceptual_grade.retry_recovery')
        callback = self._handler_boundaries[key]
        handlers = getattr(bpy.app.handlers, name)
        self.app.blender.registration.handler(handlers, callback)
        if (name, callback) not in self.handlers:
            self.handlers.append((name, callback))
        return callback

    def remove_handler(self, name, callback):
        """Remove the exact native boundary registered for a logical callback."""
        import bpy
        native = self._handler_boundaries.get((name, callback), callback)
        handlers = getattr(bpy.app.handlers, name)
        if native in handlers:
            handlers.remove(native)
        if (name, native) in self.handlers:
            self.handlers.remove((name, native))
        for key, value in tuple(self._handler_boundaries.items()):
            if key[0] == name and value is native:
                self._handler_boundaries.pop(key)

    def register(self):
        from .integration.lifecycle import live_classes
        if live_classes(strict=True):
            raise RuntimeError('A preceding product lifecycle is still registered')
        try:
            self.component('localization.i18n').register()
            nodes = self.component('graph.nodes')
            ui = self.component('ui.panel')
            diagram = self.component('ui.diagram')
            ui.prepare_linear_properties()
            nodes.register()
            ui.register()
            self.component('graph.output').register()
            self.component('integration.lifecycle').remember(nodes, ui, diagram)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.closed:
            return True
        if not self.closing:
            self.outcome_epoch += 1
        return self.component('integration.lifecycle').cleanup()
