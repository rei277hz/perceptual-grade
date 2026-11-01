"""Warp clouds from helper-rendered coverage or the baseline point renderer.

The experiment uploads only the active tone's compact coverage tiles. Cloud
coordinates and per-sample meshes stay in the helper. The baseline renderer is
retained for matched comparisons while native Metal acceptance is pending.
"""

from ..integration.components import Component

def create(_application):
    from collections import OrderedDict
    from dataclasses import dataclass
    import math
    import numpy as np
    theme = _application.component('ui.theme')
    srgb_fragment_source = _application.component('ui.shader_color').srgb_fragment_source

    POINT_SIZE=1.5
    MAX_OPACITY=.35
    CACHE_LIMIT=8
    _shader=None
    _layers=OrderedDict()
    _coverage_shader=None
    _coverage_batch=None
    _textures=OrderedDict()


    @dataclass
    class Coverage:
        # Keep publication pixels alive so identity keys cannot alias later arrays.
        pixels:object
        texture:object


    def normalize_coverage(pixels,tone,layer):
        """Convert just the displayed tile; a separate hook for upload profiling."""
        values=np.asarray(pixels)
        if values.dtype!=np.uint8 or values.ndim!=4 or values.shape[1] not in (1,4):
            raise ValueError('Warp coverage must be uint8[layer, tone, height, width]')
        return np.asarray(values[layer,0 if values.shape[1]==1 else tone],dtype=np.float32)/np.float32(255.)


    def upload_coverage(values):
        """Blender takes normalized float buffers even for an R8 texture."""
        import gpu
        height,width=values.shape
        texture=gpu.types.GPUTexture((width,height),format='R8',
            data=gpu.types.Buffer('FLOAT',(height,width),values))
        texture.filter_mode(False)
        return texture


    def texture_layer(pixels,tone,layer):
        """One upload per publication/view/tone/layer, never per redraw."""
        key=(id(pixels),int(tone),int(layer))
        item=_textures.get(key)
        if item is not None:
            _textures.move_to_end(key)
            return item.texture
        texture=upload_coverage(normalize_coverage(pixels,tone,layer))
        _textures[key]=Coverage(pixels,texture)
        while len(_textures)>CACHE_LIMIT:_textures.popitem(last=False)
        return texture


    def coverage_shader():
        import gpu
        nonlocal _coverage_shader, _coverage_batch
        if _coverage_shader is None:
            interface=gpu.types.GPUStageInterfaceInfo('pg_warp_coverage')
            interface.smooth('VEC2','textureUV')
            info=gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0,'VEC2','position');info.vertex_out(interface)
            info.push_constant('MAT4','ModelViewProjectionMatrix')
            info.push_constant('VEC4','plotRect');info.push_constant('VEC4','uvTransform')
            info.push_constant('VEC4','coverageRect');info.push_constant('VEC4','cloudColor')
            info.sampler(0,'FLOAT_2D','coverage');info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('''
    void main() {
        vec2 screen = plotRect.xy + position*plotRect.zw;
        gl_Position = ModelViewProjectionMatrix*vec4(screen, 0.0, 1.0);
        textureUV = uvTransform.xy + position*uvTransform.zw;
    }
    ''')
            info.fragment_source('''
    void main() {
        if (any(lessThan(textureUV, coverageRect.xy)) || any(greaterThanEqual(textureUV, coverageRect.zw))) discard;
        float opacity = texture(coverage, textureUV).r;
        fragColor = vec4(cloudColor.rgb, opacity*cloudColor.a);
    }
    ''')
            _coverage_shader=gpu.shader.create_from_info(info)
            from gpu_extras.batch import batch_for_shader
            _coverage_batch=batch_for_shader(_coverage_shader,'TRI_STRIP',
                {'position':((0.,0.),(1.,0.),(0.,1.),(1.,1.))})
        return _coverage_shader


    def descriptor(plot,s):
        """Native raster geometry, independent of the integer editor position."""
        x,y,width,height=map(float,plot.rect)
        origin=(x-math.floor(x),y-math.floor(y))
        return dict(width=max(1,math.ceil(origin[0]+width)),
            height=max(1,math.ceil(origin[1]+height)),
            rect=(*origin,width,height),scale=float(s),
            bounds=(float(plot.xmin),float(plot.ymin),float(plot.xmax),float(plot.ymax)))


    def choose_raster(plot,s,atlases,tone):
        """Prefer exact native views; retain the closest old raster during resize."""
        target=descriptor(plot,s)
        def distance(item):
            view=item['view']
            bounds=view['bounds'][tone]
            return (abs(view['width']-target['width'])+abs(view['height']-target['height'])+
                sum(abs(a-b) for a,b in zip(view['rect'],target['rect']))+
                100*abs(view['scale']-target['scale'])+
                100*sum(abs(a-b) for a,b in zip(bounds,target['bounds'])))
        matching=tuple(item for item in atlases if item.get('tone',tone)==tone)
        return min(matching,key=distance) if matching else None


    def draw_raster(plot,raster,tone,external):
        """Display only coverage; chart uniforms also reproject the previous view."""
        import gpu
        view=raster['view'];pixels=raster['pixels']
        sh=coverage_shader();sh.bind()
        sh.uniform_float('ModelViewProjectionMatrix',gpu.matrix.get_projection_matrix()@gpu.matrix.get_model_view_matrix())
        sh.uniform_float('plotRect',plot.rect)
        xmin,ymin,xmax,ymax=view['bounds'][tone]
        x,y,width,height=view['rect'];tw,th=view['width'],view['height']
        sx=width/(tw*(xmax-xmin));sy=height/(th*(ymax-ymin))
        sh.uniform_float('uvTransform',(x/tw+(plot.xmin-xmin)*sx,y/th+(plot.ymin-ymin)*sy,
            (plot.xmax-plot.xmin)*sx,(plot.ymax-plot.ymin)*sy))
        sh.uniform_float('coverageRect',(x/tw,y/th,(x+width)/tw,(y+height)/th))
        for layer,color in enumerate((theme.ADAPTED,theme.COMPARISON)):
            if layer>=pixels.shape[0] or (layer and external is None):break
            sh.uniform_sampler('coverage',texture_layer(pixels,tone,layer))
            sh.uniform_float('cloudColor',(*color[:3],1.))
            _coverage_batch.draw(sh)


    @dataclass
    class Layer:
        # Strong refs make identity keys safe for the lifetime of the GPU batch.
        appearance:object
        weights:object
        energy:object
        positions:object
        corners:object
        opacity:object
        negative:object
        batch:object=None


    def layer(appearance,weights,energy=None):
        """Prepare a published layer once; view changes never rebuild its mesh."""
        key=(id(appearance),id(weights),id(energy))
        existing=_layers.get(key)
        if existing is not None:
            _layers.move_to_end(key)
            return existing
        q=np.asarray(appearance).reshape(-1,2)
        w=np.asarray(weights).reshape(-1)
        valid=np.isfinite(q).all(axis=-1)&np.isfinite(w)&(w>0)
        points=np.asarray(q[valid],dtype=np.float32)
        values=np.asarray(np.clip(w[valid],0,1),dtype=np.float32)
        negative=np.zeros(len(points),dtype=np.float32) if energy is None else np.asarray(
            np.asarray(energy).reshape(-1)[valid]<0,dtype=np.float32)
        # Geometry is generated once per publication, never once per redraw.
        corners=np.array([(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)],dtype=np.float32)
        item=Layer(appearance,weights,energy,np.repeat(points,6,axis=0),
            np.tile(corners,(len(points),1)),np.repeat(values,6),np.repeat(negative,6))
        _layers[key]=item
        while len(_layers)>CACHE_LIMIT:_layers.popitem(last=False)
        return item


    def shader():
        import gpu
        nonlocal _shader
        if _shader is None:
            interface=gpu.types.GPUStageInterfaceInfo('pg_warp_samples')
            interface.smooth('VEC2','pointUV');interface.smooth('VEC2','canvasPosition')
            interface.flat('FLOAT','pointOpacity')
            interface.flat('FLOAT','pointNegative')
            info=gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0,'VEC2','opponent');info.vertex_in(1,'VEC2','corner')
            info.vertex_in(2,'FLOAT','weight');info.vertex_in(3,'FLOAT','negative');info.vertex_out(interface)
            info.push_constant('MAT4','ModelViewProjectionMatrix')
            info.push_constant('VEC4','plotRect');info.push_constant('VEC4','limits')
            info.push_constant('VEC4','pointColor');info.push_constant('FLOAT','pointRadius')
            info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('''
    void main() {
        vec2 chart = vec2(-opponent.y, opponent.x);
        vec2 screen = plotRect.xy + (chart-limits.xy)*plotRect.zw/(limits.zw-limits.xy);
        screen += corner*pointRadius*(1.0+negative);
        gl_Position = ModelViewProjectionMatrix*vec4(screen, 0.0, 1.0);
        pointUV = corner;
        canvasPosition = screen;
        pointOpacity = weight;
        pointNegative = negative;
    }
    ''')
            info.fragment_source('''
    void main() {
        if (canvasPosition.x < plotRect.x || canvasPosition.y < plotRect.y ||
            canvasPosition.x >= plotRect.x+plotRect.z || canvasPosition.y >= plotRect.y+plotRect.w)
            discard;
        float radius2 = dot(pointUV, pointUV);
        if (radius2 > 1.0 || (pointNegative > 0.5 && radius2 < 0.25)) discard;
        fragColor = vec4(pointColor.rgb, pointColor.a*pointOpacity);
    }
    ''')
            _shader=gpu.shader.create_from_info(info)
        return _shader


    def draw_layer(plot,appearance,weights,color,s,energy=None):
        import gpu
        from gpu_extras.batch import batch_for_shader
        item=layer(appearance,weights,energy)
        if not len(item.opacity):return
        sh=shader()
        if item.batch is None:
            item.batch=batch_for_shader(sh,'TRIS',dict(opponent=item.positions,corner=item.corners,
                weight=item.opacity,negative=item.negative))
        sh.bind()
        sh.uniform_float('ModelViewProjectionMatrix',gpu.matrix.get_projection_matrix()@gpu.matrix.get_model_view_matrix())
        sh.uniform_float('plotRect',plot.rect)
        sh.uniform_float('limits',(plot.xmin,plot.ymin,plot.xmax,plot.ymax))
        sh.uniform_float('pointColor',(*color[:3],MAX_OPACITY))
        sh.uniform_float('pointRadius',POINT_SIZE*s/2)
        item.batch.draw(sh)


    def draw(plot,result,external,s):
        """Draw primary blue and external red layers without changing view bounds."""
        import gpu
        if result is None and external is None:return
        blend=gpu.state.blend_get()
        try:
            gpu.state.blend_set('ALPHA')
            if result is not None and '_warp_clouds' in result:
                tone=int(result.get('_active_tone',result.get('_warp_tone',0)))
                raster=choose_raster(plot,s,result['_warp_clouds'],tone)
                if raster is not None:draw_raster(plot,raster,tone,external)
                return
            clouds = _application.component('scopes.clouds')
            # Entering Warp may retain the preceding J/C publication until its
            # first raster is ready. Never turn those old arrays into UI meshes.
            if clouds.enabled():return
            if result is not None:
                draw_layer(plot,result['bounded_saturation'],result['weights'],theme.ADAPTED,s,
                           energy=result['after_energy'])
            if external is not None:
                draw_layer(plot,external['signal']['bounded_saturation'],external['weights'],theme.COMPARISON,s,
                           energy=external['signal']['energy'])
        finally:gpu.state.blend_set(blend)


    def clear():
        nonlocal _shader, _coverage_shader, _coverage_batch, _density_shader, _density_batch, _histogram_shader, _histogram_batch
        _layers.clear();_textures.clear()
        _density_textures.clear();_histogram_textures.clear()
        _shader=_coverage_shader=_coverage_batch=None
        _density_shader=_density_batch=_histogram_shader=_histogram_batch=None


    _density_shader=None
    _density_batch=None
    _density_textures=OrderedDict()
    _histogram_shader=None
    _histogram_batch=None
    _histogram_textures=OrderedDict()


    def normalize_density(levels):
        values=np.asarray(levels)
        if values.dtype!=np.uint8 or values.ndim!=2:raise ValueError('Density levels must be a uint8 image')
        return np.asarray(values,dtype=np.float32)/np.float32(255.)


    def draw_density(rect,levels,color,alpha=1.):
        """Upload quantized helper density once and draw one GPU quad."""
        import gpu
        from gpu_extras.batch import batch_for_shader
        nonlocal _density_shader, _density_batch
        item=_density_textures.get(id(levels))
        if item is None:
            item=Coverage(levels,upload_coverage(normalize_density(levels)))
            _density_textures[id(levels)]=item
            while len(_density_textures)>CACHE_LIMIT:_density_textures.popitem(last=False)
        if _density_shader is None:
            interface=gpu.types.GPUStageInterfaceInfo('pg_scope_density')
            interface.smooth('VEC2','uv')
            info=gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0,'VEC2','position');info.vertex_out(interface)
            info.push_constant('MAT4','ModelViewProjectionMatrix');info.push_constant('VEC4','plotRect')
            info.push_constant('VEC4','pointColor');info.sampler(0,'FLOAT_2D','density')
            info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('void main(){ uv=position; gl_Position=ModelViewProjectionMatrix*vec4(plotRect.xy+position*plotRect.zw,0.,1.); }')
            info.fragment_source('void main(){ float level=round(texture(density,uv).r*255.); if(level<.5) discard; fragColor=vec4(pointColor.rgb,pointColor.a*(.12+.88*level/7.)); }')
            _density_shader=gpu.shader.create_from_info(info)
            _density_batch=batch_for_shader(_density_shader,'TRI_STRIP',{'position':((0.,0.),(1.,0.),(0.,1.),(1.,1.))})
        sh=_density_shader;sh.bind()
        sh.uniform_float('ModelViewProjectionMatrix',gpu.matrix.get_projection_matrix()@gpu.matrix.get_model_view_matrix())
        sh.uniform_float('plotRect',rect);sh.uniform_float('pointColor',(*color[:3],alpha))
        sh.uniform_sampler('density',item.texture);_density_batch.draw(sh)


    def draw_histograms(plot,result,drivers,selected):
        """Cache tiny input histograms and instance bars without CPU triangles."""
        import gpu
        from gpu_extras.batch import batch_for_shader
        nonlocal _histogram_shader, _histogram_batch
        values=result['histograms'];key=tuple(id(values[name]) for name in drivers)
        cached=_histogram_textures.get(key)
        if cached is None:
            arrays=tuple(values[name] for name in drivers)
            textures=[]
            for source in arrays:
                array=np.asarray(source,dtype=np.float32).reshape(1,-1)
                texture=gpu.types.GPUTexture((array.size,1),format='R32F',
                    data=gpu.types.Buffer('FLOAT',array.shape,array))
                texture.filter_mode(False);textures.append(texture)
            cached=(arrays,tuple(textures),max(.01,max(float(np.max(np.abs(v))) for v in arrays)))
            _histogram_textures[key]=cached
            while len(_histogram_textures)>CACHE_LIMIT:_histogram_textures.popitem(last=False)
        if _histogram_shader is None:
            info=gpu.types.GPUShaderCreateInfo();info.vertex_in(0,'VEC2','position')
            info.push_constant('MAT4','ModelViewProjectionMatrix');info.push_constant('VEC4','plotRect')
            # Group vectors before scalars to keep srgbTarget within the 128-byte budget.
            info.push_constant('VEC4','pointColor')
            info.push_constant('FLOAT','maximum');info.push_constant('INT','binCount')
            info.sampler(0,'FLOAT_2D','histogram')
            info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('''void main(){
     float height=texelFetch(histogram,ivec2(gl_InstanceID,0),0).r/maximum*.78;
     vec2 p=vec2((float(gl_InstanceID)+position.x)/float(binCount),position.y*height);
     gl_Position=ModelViewProjectionMatrix*vec4(plotRect.xy+p*plotRect.zw,0.,1.);
    }''')
            srgb_fragment_source(info,'void main(){ fragColor=pg_srgb_to_framebuffer(pointColor); }')
            _histogram_shader=gpu.shader.create_from_info(info)
            _histogram_batch=batch_for_shader(_histogram_shader,'TRIS',
                {'position':((0.,0.),(1.,0.),(1.,1.),(0.,0.),(1.,1.),(0.,1.))})
        sh=_histogram_shader;sh.bind()
        sh.uniform_float('ModelViewProjectionMatrix',gpu.matrix.get_projection_matrix()@gpu.matrix.get_model_view_matrix())
        sh.uniform_float('plotRect',plot.rect);sh.uniform_float('maximum',cached[2])
        for index in sorted(range(len(drivers)),key=lambda i:drivers[i]==selected):
            name=drivers[index];color=theme.ADAPTED if name=='low' else theme.ORANGE
            sh.uniform_int('binCount',len(cached[0][index]))
            sh.uniform_float('pointColor',(*color[:3],.26 if name==selected else .055))
            sh.uniform_sampler('histogram',cached[1][index])
            _histogram_batch.draw_instanced(sh,instance_count=len(cached[0][index]))

    def _set_export_CACHE_LIMIT(value):
        nonlocal CACHE_LIMIT
        CACHE_LIMIT = value

    def _set_export_MAX_OPACITY(value):
        nonlocal MAX_OPACITY
        MAX_OPACITY = value

    def _set_export_POINT_SIZE(value):
        nonlocal POINT_SIZE
        POINT_SIZE = value

    def _set_export__coverage_batch(value):
        nonlocal _coverage_batch
        _coverage_batch = value

    def _set_export__coverage_shader(value):
        nonlocal _coverage_shader
        _coverage_shader = value

    def _set_export__density_batch(value):
        nonlocal _density_batch
        _density_batch = value

    def _set_export__density_shader(value):
        nonlocal _density_shader
        _density_shader = value

    def _set_export__density_textures(value):
        nonlocal _density_textures
        _density_textures = value

    def _set_export__histogram_batch(value):
        nonlocal _histogram_batch
        _histogram_batch = value

    def _set_export__histogram_shader(value):
        nonlocal _histogram_shader
        _histogram_shader = value

    def _set_export__histogram_textures(value):
        nonlocal _histogram_textures
        _histogram_textures = value

    def _set_export__layers(value):
        nonlocal _layers
        _layers = value

    def _set_export__shader(value):
        nonlocal _shader
        _shader = value

    def _set_export__textures(value):
        nonlocal _textures
        _textures = value

    def _set_export_Coverage(value):
        nonlocal Coverage
        Coverage = value

    def _set_export_normalize_coverage(value):
        nonlocal normalize_coverage
        normalize_coverage = value

    def _set_export_upload_coverage(value):
        nonlocal upload_coverage
        upload_coverage = value

    def _set_export_texture_layer(value):
        nonlocal texture_layer
        texture_layer = value

    def _set_export_coverage_shader(value):
        nonlocal coverage_shader
        coverage_shader = value

    def _set_export_descriptor(value):
        nonlocal descriptor
        descriptor = value

    def _set_export_choose_raster(value):
        nonlocal choose_raster
        choose_raster = value

    def _set_export_draw_raster(value):
        nonlocal draw_raster
        draw_raster = value

    def _set_export_Layer(value):
        nonlocal Layer
        Layer = value

    def _set_export_layer(value):
        nonlocal layer
        layer = value

    def _set_export_shader(value):
        nonlocal shader
        shader = value

    def _set_export_draw_layer(value):
        nonlocal draw_layer
        draw_layer = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_normalize_density(value):
        nonlocal normalize_density
        normalize_density = value

    def _set_export_draw_density(value):
        nonlocal draw_density
        draw_density = value

    def _set_export_draw_histograms(value):
        nonlocal draw_histograms
        draw_histograms = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CACHE_LIMIT': lambda: CACHE_LIMIT,
        'Coverage': lambda: Coverage,
        'Layer': lambda: Layer,
        'MAX_OPACITY': lambda: MAX_OPACITY,
        'OrderedDict': lambda: OrderedDict,
        'POINT_SIZE': lambda: POINT_SIZE,
        '_coverage_batch': lambda: _coverage_batch,
        '_coverage_shader': lambda: _coverage_shader,
        '_density_batch': lambda: _density_batch,
        '_density_shader': lambda: _density_shader,
        '_density_textures': lambda: _density_textures,
        '_histogram_batch': lambda: _histogram_batch,
        '_histogram_shader': lambda: _histogram_shader,
        '_histogram_textures': lambda: _histogram_textures,
        '_layers': lambda: _layers,
        '_shader': lambda: _shader,
        '_textures': lambda: _textures,
        'choose_raster': lambda: choose_raster,
        'clear': lambda: clear,
        'coverage_shader': lambda: coverage_shader,
        'dataclass': lambda: dataclass,
        'descriptor': lambda: descriptor,
        'draw': lambda: draw,
        'draw_density': lambda: draw_density,
        'draw_histograms': lambda: draw_histograms,
        'draw_layer': lambda: draw_layer,
        'draw_raster': lambda: draw_raster,
        'layer': lambda: layer,
        'math': lambda: math,
        'normalize_coverage': lambda: normalize_coverage,
        'normalize_density': lambda: normalize_density,
        'np': lambda: np,
        'shader': lambda: shader,
        'srgb_fragment_source': lambda: srgb_fragment_source,
        'texture_layer': lambda: texture_layer,
        'theme': lambda: theme,
        'upload_coverage': lambda: upload_coverage,
    }, writers={
        'CACHE_LIMIT': _set_export_CACHE_LIMIT,
        'MAX_OPACITY': _set_export_MAX_OPACITY,
        'POINT_SIZE': _set_export_POINT_SIZE,
        '_coverage_batch': _set_export__coverage_batch,
        '_coverage_shader': _set_export__coverage_shader,
        '_density_batch': _set_export__density_batch,
        '_density_shader': _set_export__density_shader,
        '_density_textures': _set_export__density_textures,
        '_histogram_batch': _set_export__histogram_batch,
        '_histogram_shader': _set_export__histogram_shader,
        '_histogram_textures': _set_export__histogram_textures,
        '_layers': _set_export__layers,
        '_shader': _set_export__shader,
        '_textures': _set_export__textures,
    'Coverage': _set_export_Coverage, 'normalize_coverage': _set_export_normalize_coverage, 'upload_coverage': _set_export_upload_coverage, 'texture_layer': _set_export_texture_layer, 'coverage_shader': _set_export_coverage_shader, 'descriptor': _set_export_descriptor, 'choose_raster': _set_export_choose_raster, 'draw_raster': _set_export_draw_raster, 'Layer': _set_export_Layer, 'layer': _set_export_layer, 'shader': _set_export_shader, 'draw_layer': _set_export_draw_layer, 'draw': _set_export_draw, 'clear': _set_export_clear, 'normalize_density': _set_export_normalize_density, 'draw_density': _set_export_draw_density, 'draw_histograms': _set_export_draw_histograms})
