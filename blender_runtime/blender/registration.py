"""App-owned exact registration handles and exhaustive ordered cleanup."""


class Registration:
    def __init__(self, app):
        self.app = app
        self._cleanup = []

    def __bool__(self):
        return bool(self._cleanup)

    def defer(self, callback):
        self.app.require_open()
        if not callable(callback):
            raise TypeError('Cleanup requires a callable')
        self._cleanup.append(callback)

    def register_class(self, cls):
        self.app.require_open()
        import bpy
        bpy.utils.register_class(cls)
        self.defer(lambda: bpy.utils.unregister_class(cls))

    def property(self, owner, name, value):
        self.app.require_open()
        if hasattr(owner, name):
            raise RuntimeError('Blender property is already registered: ' + name)
        setattr(owner, name, value)
        self.defer(lambda: delattr(owner, name))

    def handler(self, handlers, callback):
        self.app.require_open()
        if callback not in handlers:
            handlers.append(callback)
            self.defer(lambda: handlers.remove(callback) if callback in handlers else None)

    def timer(self, callback, **settings):
        self.app.require_open()
        import bpy
        if not bpy.app.timers.is_registered(callback):
            bpy.app.timers.register(callback, **settings)
            self.defer(lambda: bpy.app.timers.unregister(callback)
                       if bpy.app.timers.is_registered(callback) else None)

    def close(self):
        errors, pending = [], []
        while self._cleanup:
            callback = self._cleanup.pop()
            try:
                callback()
            except Exception as error:
                errors.append(error)
                pending.append(callback)
        if errors:
            # Retain exact failed resources for a later cleanup attempt. A
            # second close must not silently bless an abandoned registration.
            self._cleanup.extend(reversed(pending))
            raise ExceptionGroup('Blender registration cleanup failed', errors)
