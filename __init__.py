"""Independent Blender extension entrypoint; dependencies are privately bundled."""

RELEASE = '1.23.2'

def register():
    from .perceptual_grade_blender import register as enable
    enable()


def unregister():
    from .perceptual_grade_blender import unregister as disable
    disable()
