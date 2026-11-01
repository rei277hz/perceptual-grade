"""Perceptual Grade worker operations; the shared worker knows no models."""


def dispatch(operation, inputs, context):
    if operation == 'scope_image_read':
        from .handlers import images
        return images.handle(inputs, context)
    if operation == 'scope_capture':
        from .handlers import native_capture
        return native_capture.handle(inputs, context)
    if operation == 'compositor_bake':
        from .handlers import native_capture
        return native_capture.handle(dict(inputs, _bake=True), context)
    if operation == 'shader_bake':
        from .handlers import shader_bake
        return shader_bake.handle(inputs, context)
    if operation in ('lut_bake', 'pack_lut', 'gpu_probe'):
        from .handlers import bake
        if operation == 'gpu_probe':
            return bake.handle_probe(inputs, context)
        return (bake.handle if operation == 'lut_bake' else bake.pack)(inputs, context)
    if operation in ('scope_linear', 'scope_full', 'scope_warp', 'scope_register', 'scope_update', 'scope_probe'):
        from .handlers import scopes
        return getattr(scopes, operation.removeprefix('scope_'))(inputs, context)
    raise ValueError('Unknown execution operation: ' + operation)
