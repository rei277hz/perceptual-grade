"""Fixed palette; independent of the user's current Blender theme.

Logo accents: https://www.blender.org/about/logo/ (#e87d0d, #265787).
Panel/text/axis colors: Blender 5.2 factory Dark theme. Additional neutrals
and complementary direction colors are chosen for this chart's readability.
No Blender logo artwork is included.
"""

from ..integration.components import Component

def create(_application):


    def rgb(value):
        return tuple(int(value[i:i+2], 16)/255 for i in (0, 2, 4))+(1.,)


    ORANGE = rgb('e87d0d')
    BLUE = rgb('265787')
    PANEL = rgb('3d3d3d')
    HEADER = rgb('303030')
    PLOT = rgb('282828')
    OUTLINE = rgb('575757')
    GRID = rgb('404040')
    TEXT = rgb('e6e6e6')
    MUTED = rgb('aaaaaa')
    REFERENCE = rgb('9b9b9b')
    ADAPTED = rgb('87a3bd')  # lightened brand blue for small text on dark panels
    GRADED = rgb('e6e6e6')
    COLORS = {'cr': (rgb('ff3352'), rgb('4bc9cc')),
              'yb': (rgb('2890ff'), rgb('f0cf5a'))}

    COMPARISON = rgb('f46b70')  # red scope overlay over the primary blue trace

    def _set_export_ADAPTED(value):
        nonlocal ADAPTED
        ADAPTED = value

    def _set_export_BLUE(value):
        nonlocal BLUE
        BLUE = value

    def _set_export_COLORS(value):
        nonlocal COLORS
        COLORS = value

    def _set_export_COMPARISON(value):
        nonlocal COMPARISON
        COMPARISON = value

    def _set_export_GRADED(value):
        nonlocal GRADED
        GRADED = value

    def _set_export_GRID(value):
        nonlocal GRID
        GRID = value

    def _set_export_HEADER(value):
        nonlocal HEADER
        HEADER = value

    def _set_export_MUTED(value):
        nonlocal MUTED
        MUTED = value

    def _set_export_ORANGE(value):
        nonlocal ORANGE
        ORANGE = value

    def _set_export_OUTLINE(value):
        nonlocal OUTLINE
        OUTLINE = value

    def _set_export_PANEL(value):
        nonlocal PANEL
        PANEL = value

    def _set_export_PLOT(value):
        nonlocal PLOT
        PLOT = value

    def _set_export_REFERENCE(value):
        nonlocal REFERENCE
        REFERENCE = value

    def _set_export_TEXT(value):
        nonlocal TEXT
        TEXT = value

    def _set_export_rgb(value):
        nonlocal rgb
        rgb = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ADAPTED': lambda: ADAPTED,
        'BLUE': lambda: BLUE,
        'COLORS': lambda: COLORS,
        'COMPARISON': lambda: COMPARISON,
        'GRADED': lambda: GRADED,
        'GRID': lambda: GRID,
        'HEADER': lambda: HEADER,
        'MUTED': lambda: MUTED,
        'ORANGE': lambda: ORANGE,
        'OUTLINE': lambda: OUTLINE,
        'PANEL': lambda: PANEL,
        'PLOT': lambda: PLOT,
        'REFERENCE': lambda: REFERENCE,
        'TEXT': lambda: TEXT,
        'rgb': lambda: rgb,
    }, writers={
        'ADAPTED': _set_export_ADAPTED,
        'BLUE': _set_export_BLUE,
        'COLORS': _set_export_COLORS,
        'COMPARISON': _set_export_COMPARISON,
        'GRADED': _set_export_GRADED,
        'GRID': _set_export_GRID,
        'HEADER': _set_export_HEADER,
        'MUTED': _set_export_MUTED,
        'ORANGE': _set_export_ORANGE,
        'OUTLINE': _set_export_OUTLINE,
        'PANEL': _set_export_PANEL,
        'PLOT': _set_export_PLOT,
        'REFERENCE': _set_export_REFERENCE,
        'TEXT': _set_export_TEXT,
    'rgb': _set_export_rgb})
