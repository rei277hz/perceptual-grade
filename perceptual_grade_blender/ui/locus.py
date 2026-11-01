"""Draw the bundled, human-viewable CIE 1931 xy locus PNG directly."""

from ..integration.components import Component

def create(_application):
    from pathlib import Path
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve

    ASSET=Path(__file__).parent.parent/'assets'/'cie_1931_xy_locus.png'
    LOW=0.
    HIGH=.9
    SIZE=1024
    _image=None
    _texture=None
    _shader=None
    _batch=None
    _rect=None


    def draw(chart):
        import bpy
        import gpu
        from gpu_extras.batch import batch_for_shader
        nonlocal _image, _texture, _shader, _batch, _rect
        if _texture is None:
            image=bpy.data.images.load(str(ASSET),check_existing=True)
            image.colorspace_settings.name='Non-Color'
            image.alpha_mode='STRAIGHT'
            _image=retain(image,'locus.image')
            _texture=gpu.texture.from_image(image)
        if _shader is None:_shader=gpu.shader.from_builtin('IMAGE')
        rect=(chart.left,chart.bottom,chart.size,chart.low,chart.high)
        if _batch is None or rect!=_rect:
            x,y,n=rect[:3]
            lo=(chart.low-LOW)/(HIGH-LOW);hi=(chart.high-LOW)/(HIGH-LOW)
            _batch=batch_for_shader(_shader,'TRI_FAN',
                {'pos':[(x,y),(x+n,y),(x+n,y+n),(x,y+n)],
                 'texCoord':[(lo,lo),(hi,lo),(hi,hi),(lo,hi)]})
            _rect=rect
        # Text/other draw helpers can leave blending disabled. Explicit alpha
        # compositing keeps the PNG's transparent pixels clear over the chart.
        blend=gpu.state.blend_get()
        try:
            gpu.state.blend_set('ALPHA')
            _shader.bind();_shader.uniform_sampler('image',_texture);_batch.draw(_shader)
        finally:gpu.state.blend_set(blend)


    def clear():
        nonlocal _image, _texture, _shader, _batch, _rect
        _image=_texture=_shader=_batch=_rect=None

    def _set_export_ASSET(value):
        nonlocal ASSET
        ASSET = value

    def _set_export_HIGH(value):
        nonlocal HIGH
        HIGH = value

    def _set_export_LOW(value):
        nonlocal LOW
        LOW = value

    def _set_export_SIZE(value):
        nonlocal SIZE
        SIZE = value

    def _set_export__batch(value):
        nonlocal _batch
        _batch = value

    def _set_export__image(value):
        nonlocal _image
        _image = value

    def _set_export__rect(value):
        nonlocal _rect
        _rect = value

    def _set_export__shader(value):
        nonlocal _shader
        _shader = value

    def _set_export__texture(value):
        nonlocal _texture
        _texture = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ASSET': lambda: ASSET,
        'HIGH': lambda: HIGH,
        'LOW': lambda: LOW,
        'Path': lambda: Path,
        'SIZE': lambda: SIZE,
        '_batch': lambda: _batch,
        '_image': lambda: _image,
        '_rect': lambda: _rect,
        '_shader': lambda: _shader,
        '_texture': lambda: _texture,
        'clear': lambda: clear,
        'draw': lambda: draw,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
    }, writers={
        'ASSET': _set_export_ASSET,
        'HIGH': _set_export_HIGH,
        'LOW': _set_export_LOW,
        'SIZE': _set_export_SIZE,
        '_batch': _set_export__batch,
        '_image': _set_export__image,
        '_rect': _set_export__rect,
        '_shader': _set_export__shader,
        '_texture': _set_export__texture,
    'draw': _set_export_draw, 'clear': _set_export_clear})
