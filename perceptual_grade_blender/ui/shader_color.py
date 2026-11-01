"""Match Blender's built-in shaders when drawing authored sRGB UI colors."""

from ..integration.components import Component

def create(_application):

    # Equivalent to Blender's gpu_shader_colorspace_lib.glsl. An sRGB framebuffer
    # encodes shader output, so decode our display colors only for that target.
    _SRGB_TO_FRAMEBUFFER = '''
    vec4 pg_srgb_to_framebuffer(vec4 color){
        if(!srgbTarget)return color;
        vec3 c=max(color.rgb,vec3(0.0));
        vec3 linear=mix(c/12.92,pow((c+0.055)/1.055,vec3(2.4)),
                        step(vec3(0.04045),c));
        return vec4(linear,color.a);
    }
    '''


    def srgb_fragment_source(info, source):
        """Declare the target flag and add the shared fragment color conversion."""
        # Blender sets this reserved uniform on bind, including framebuffer changes.
        info.push_constant('BOOL', 'srgbTarget')
        info.fragment_source(_SRGB_TO_FRAMEBUFFER + source)

    def _set_export__SRGB_TO_FRAMEBUFFER(value):
        nonlocal _SRGB_TO_FRAMEBUFFER
        _SRGB_TO_FRAMEBUFFER = value

    def _set_export_srgb_fragment_source(value):
        nonlocal srgb_fragment_source
        srgb_fragment_source = value

    return Component(readers={
        "__name__": lambda: __name__,
        '_SRGB_TO_FRAMEBUFFER': lambda: _SRGB_TO_FRAMEBUFFER,
        'srgb_fragment_source': lambda: srgb_fragment_source,
    }, writers={
        '_SRGB_TO_FRAMEBUFFER': _set_export__SRGB_TO_FRAMEBUFFER,
    'srgb_fragment_source': _set_export_srgb_fragment_source})
