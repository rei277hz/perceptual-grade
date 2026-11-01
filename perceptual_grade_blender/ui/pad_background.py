"""Cached a/b backdrop and input hue histogram, rendered by one GPU quad.

The immutable asset uses a shared a/b grid at H=45 and a saturation-preserving
remap to H=65. Histogram sectors reveal the brighter appearance image. Only
48 reduced counts arrive from the helper; no point arrays or CPU meshes enter
the UI. Screen coordinates are (-b,a).
"""

from ..integration.components import Component

def create(_application):
    from collections import OrderedDict
    from functools import lru_cache
    from pathlib import Path
    import numpy as np
    tr = _application.component('localization.i18n').tr
    load_rgba8 = _application.component('ui.png_assets').load_rgba8
    srgb_fragment_source = _application.component('ui.shader_color').srgb_fragment_source

    _shader=None
    _batch=None
    _background=None
    _sectors=OrderedDict()
    _EMPTY=np.zeros(48,dtype=np.float32)


    @lru_cache(maxsize=1)
    def asset():
        pixels=load_rgba8(Path(__file__).parent.parent/'assets'/'ab_background.png')
        n=pixels.shape[1]
        if pixels.shape!=(n*2,n,4):raise ValueError(tr('error.pad_asset_shape'))
        return dict(dark=pixels[:n].astype(np.float32)/255,
                    bright=pixels[n:].astype(np.float32)/255)


    def _upload_sectors(values):
        import gpu
        texture=gpu.types.GPUTexture((48,1),format='R32F',
            data=gpu.types.Buffer('FLOAT',(1,48),values))
        texture.filter_mode(False)
        return texture


    def sector_texture(source=None):
        """One tiny upload per publication; retain the source to prevent id reuse."""
        source=_EMPTY if source is None else source
        key=id(source)
        if key not in _sectors:
            values=np.asarray(source,dtype=np.float32)
            if values.shape!=(48,) or not np.isfinite(values).all() or np.any(values<0):
                raise ValueError('Pad sectors must be 48 finite nonnegative counts')
            _sectors[key]=(source,_upload_sectors(values),max(float(values.max()),1e-20))
            while len(_sectors)>12:_sectors.popitem(last=False)
        _sectors.move_to_end(key)
        return _sectors[key]


    def background_textures():
        import gpu
        nonlocal _background
        if _background is None:
            values=asset();textures=[]
            for name in ('dark','bright'):
                pixels=values[name];height,width=pixels.shape[:2]
                texture=gpu.types.GPUTexture((width,height),format='RGBA8',
                    data=gpu.types.Buffer('FLOAT',pixels.shape,pixels))
                texture.filter_mode(True);textures.append(texture)
            _background=tuple(textures)
        return _background


    def shader():
        import gpu
        nonlocal _shader, _batch
        if _shader is None:
            from gpu_extras.batch import batch_for_shader
            interface=gpu.types.GPUStageInterfaceInfo('pg_pad_background')
            interface.smooth('VEC2','uv')
            info=gpu.types.GPUShaderCreateInfo();info.vertex_in(0,'VEC2','position');info.vertex_out(interface)
            info.push_constant('MAT4','ModelViewProjectionMatrix');info.push_constant('VEC4','plotRect')
            info.push_constant('FLOAT','sectorMaximum')
            for slot,name in enumerate(('darkImage','brightImage','sectorCounts')):info.sampler(slot,'FLOAT_2D',name)
            info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('''void main(){
        uv=position;
        gl_Position=ModelViewProjectionMatrix*vec4(plotRect.xy+position*plotRect.zw,0.0,1.0);
    }''')
            srgb_fragment_source(info,'''void main(){
        vec2 coordinate=uv*2.0-1.0;
        float angle=any(notEqual(coordinate,vec2(0.0)))?atan(-coordinate.x,coordinate.y):0.0;
        int sector=clamp(int(floor((angle+3.141592653589793)*(48.0/(2.0*3.141592653589793)))),0,47);
        float count=texelFetch(sectorCounts,ivec2(sector,0),0).r;
        float radius=.88*sqrt(max(count,0.0)/sectorMaximum);
        float mask=count>0.0&&length(coordinate)<=radius?1.0:0.0;
        vec2 size=vec2(textureSize(darkImage,0));
        vec2 imageUV=(vec2(.5)+uv*(size-1.0))/size;
        fragColor=pg_srgb_to_framebuffer(mix(texture(darkImage,imageUV),texture(brightImage,imageUV),mask));
    }''')
            _shader=gpu.shader.create_from_info(info)
            _batch=batch_for_shader(_shader,'TRI_STRIP',
                {'position':((0.,0.),(1.,0.),(0.,1.),(1.,1.))})
        return _shader


    def draw(plot,result=None,neutral=False):
        import gpu
        source=None if neutral or result is None else result.get('sectors')
        _,sectors,maximum=sector_texture(source)
        dark,bright=background_textures();sh=shader();sh.bind()
        sh.uniform_float('ModelViewProjectionMatrix',gpu.matrix.get_projection_matrix()@gpu.matrix.get_model_view_matrix())
        sh.uniform_float('plotRect',plot.rect);sh.uniform_float('sectorMaximum',maximum)
        sh.uniform_sampler('darkImage',dark);sh.uniform_sampler('brightImage',bright)
        sh.uniform_sampler('sectorCounts',sectors);_batch.draw(sh)


    def clear():
        nonlocal _shader, _batch, _background
        _shader=_batch=_background=None
        _sectors.clear();asset.cache_clear()

    def _set_export__EMPTY(value):
        nonlocal _EMPTY
        _EMPTY = value

    def _set_export__background(value):
        nonlocal _background
        _background = value

    def _set_export__batch(value):
        nonlocal _batch
        _batch = value

    def _set_export__sectors(value):
        nonlocal _sectors
        _sectors = value

    def _set_export__shader(value):
        nonlocal _shader
        _shader = value

    def _set_export_asset(value):
        nonlocal asset
        asset = value

    def _set_export__upload_sectors(value):
        nonlocal _upload_sectors
        _upload_sectors = value

    def _set_export_sector_texture(value):
        nonlocal sector_texture
        sector_texture = value

    def _set_export_background_textures(value):
        nonlocal background_textures
        background_textures = value

    def _set_export_shader(value):
        nonlocal shader
        shader = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'OrderedDict': lambda: OrderedDict,
        'Path': lambda: Path,
        '_EMPTY': lambda: _EMPTY,
        '_background': lambda: _background,
        '_batch': lambda: _batch,
        '_sectors': lambda: _sectors,
        '_shader': lambda: _shader,
        '_upload_sectors': lambda: _upload_sectors,
        'asset': lambda: asset,
        'background_textures': lambda: background_textures,
        'clear': lambda: clear,
        'draw': lambda: draw,
        'load_rgba8': lambda: load_rgba8,
        'lru_cache': lambda: lru_cache,
        'np': lambda: np,
        'sector_texture': lambda: sector_texture,
        'shader': lambda: shader,
        'srgb_fragment_source': lambda: srgb_fragment_source,
        'tr': lambda: tr,
    }, writers={
        '_EMPTY': _set_export__EMPTY,
        '_background': _set_export__background,
        '_batch': _set_export__batch,
        '_sectors': _set_export__sectors,
        '_shader': _set_export__shader,
    'asset': _set_export_asset, '_upload_sectors': _set_export__upload_sectors, 'sector_texture': _set_export_sector_texture, 'background_textures': _set_export_background_textures, 'shader': _set_export_shader, 'draw': _set_export_draw, 'clear': _set_export_clear})
