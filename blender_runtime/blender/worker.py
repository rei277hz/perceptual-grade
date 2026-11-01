"""Blender-specific worker initialization and diagnostics."""


def initialize(settings):
    import bpy
    backend = settings.get('backend', '')
    if backend:
        try:
            bpy.context.preferences.system.gpu_backend = backend
        except (TypeError, ValueError):
            pass
    return {'blender': list(bpy.app.version), 'backend': backend}


def diagnostic(gpu_role):
    import bpy
    value = {'backend': getattr(bpy.context.preferences.system, 'gpu_backend', ''), 'device': ''}
    if gpu_role:
        try:
            import gpu
            value['device'] = gpu.platform.renderer_get()
        except Exception:
            pass
    return value
