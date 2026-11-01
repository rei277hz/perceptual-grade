"""Foreground source-only Image decoding and immutable, bounded scope samples.

Descriptors contain values, never RNA handles. Private datablocks are confined
inside one guarded read; caches retain only arrays and value identities.
"""
from dataclasses import dataclass, replace
from ..integration.components import Component


class SourcePreparing(RuntimeError):
    """An isolated stored-source decode has not physically completed yet."""


@dataclass(frozen=True)
class FrameSettings:
    start: int = 1
    duration: int = 1
    offset: int = 0
    cyclic: bool = False

    def resolve(self, scene_frame):
        duration=max(1,int(self.duration))
        value=int(scene_frame)-int(self.start)+1
        if self.cyclic:value=(value-1)%duration+1
        return min(duration,max(0,value))+int(self.offset)


@dataclass(frozen=True)
class SourceDescriptor:
    lifetime: tuple
    name: str
    source: str
    image_type: str
    filepath: str
    colorspace: str
    color_context: tuple
    alpha_mode: str
    source_kind: str
    premultiply: bool
    frame: int
    frame_settings: FrameSettings
    layer: object
    pass_name: object
    view: object
    tile_numbers: tuple
    multiview: bool
    view_settings: tuple
    stereo_settings: tuple
    channel: str
    metadata: tuple

    @property
    def animated(self):return self.source in {'MOVIE','SEQUENCE'}

    @property
    def continuity(self):
        return ('image-source-v2',self.lifetime,self.source,self.image_type,
                self.filepath,self.colorspace,self.color_context,self.alpha_mode,self.source_kind,
                self.premultiply,(self.frame_settings.start,self.frame_settings.duration,
                self.frame_settings.offset,self.frame_settings.cyclic),self.layer,self.pass_name,
                self.view,self.tile_numbers,self.multiview,self.view_settings,self.stereo_settings,self.channel)

    @property
    def key(self):return self.continuity+(self.frame if self.animated else None,self.metadata)


@dataclass(frozen=True,eq=False)
class SampleResult:
    rgba: object
    layout: object
    key: tuple
    revision: int
    label: str
    incomplete: bool = False


def create(_application, *, background_movies=False):
    from collections import OrderedDict
    from contextlib import contextmanager
    import os
    import re
    import tempfile
    import shutil
    import hashlib
    import time
    import bpy
    import numpy as np
    from ...perceptual_grade_core.scopes.samples import (
        immutable,ordinary_layout,tiled_layout,sample_rgba)

    guard=_application.component('graph.output_guard')
    exr=_application.component('scopes.exr')
    cache=OrderedDict()
    pending=OrderedDict()
    retiring={}
    checked={}
    generation=0
    sequence=0
    sampler_shader=None
    PROBE_INTERVAL=.5
    diagnostic_state=dict(enabled=False,reads=0,decodes=0,decode_seconds=0.,events=[])

    def enable_diagnostics(enabled=True):
        diagnostic_state.update(enabled=bool(enabled),reads=0,decodes=0,decode_seconds=0.,events=[])

    def diagnostics():
        return dict(diagnostic_state,events=tuple(diagnostic_state['events']))

    def _stat(path):
        try:
            item=os.stat(path)
            return item.st_size,item.st_mtime_ns,item.st_ctime_ns
        except OSError:return None

    def _frame_settings(value):
        if value is None:return FrameSettings()
        if isinstance(value,FrameSettings):return value
        fields={'start':'frame_start','duration':'frame_duration',
                'offset':'frame_offset','cyclic':'use_cyclic'}
        if isinstance(value,(tuple,list)):return FrameSettings(*value)
        return FrameSettings(**{key:value.get(key,value.get(other,getattr(FrameSettings(),key)))
                                for key,other in fields.items()})

    def describe(image,*,source_kind='shader',frame_settings=None,scene_frame=1,
                 layer=None,pass_name='Combined',view=None,premultiply=False,scene=None,channel='COLOR'):
        settings=_frame_settings(frame_settings)
        path=bpy.path.abspath(image.filepath,library=image.library) if image.filepath else ''
        source=image.source
        # Blender's static multiview getters use a null ImageUser and can
        # reload the file, dropping resident painted-eye buffers. Dimensions
        # for these sources belong to the private decoder, not the user Image.
        # RNA size/channels/is_float/frame_duration may initialize or replace an
        # Image buffer. Discovery reads stored metadata only, including while
        # the foreground compositor holds an output-reader lease.
        dimensions=()
        if source not in {'MOVIE','SEQUENCE'}:settings=FrameSettings()
        # Unloaded EXRs report IMAGE until native decoding discovers their
        # layers/views. Preserve selection without triggering that discovery.
        may_have_layers=image.type=='MULTILAYER' or os.path.splitext(path)[1].lower()=='.exr'
        if not may_have_layers and source not in {'MOVIE','SEQUENCE'}:layer=None;pass_name='Combined'
        if not image.use_multiview and not may_have_layers and source not in {'MOVIE','SEQUENCE'}:view=None
        elif view in (0,'','ALL'):view=None
        tiles=tuple(sorted(tile.number for tile in image.tiles)) if source=='TILED' else ()
        packed=tuple((item.tile_number,item.view,item.packed_file.size)
                     for item in image.packed_files)
        scene=scene or bpy.context.scene
        views=tuple((item.name,item.file_suffix) for item in scene.render.views if item.use) if scene is not None and image.use_multiview else ()
        stereo=image.stereo_3d_format
        stereo_settings=(image.views_format,stereo.display_mode,stereo.anaglyph_type,
            stereo.interlace_type,stereo.use_interlace_swap,stereo.use_sidebyside_crosseyed,
            stereo.use_squeezed_frame) if image.use_multiview else ()
        frame=settings.resolve(scene_frame)
        if source=='MOVIE':frame=max(1,frame)
        elif source=='SEQUENCE':frame=max(0,frame)
        resource_path=_sequence_filepath(path,frame) if source=='SEQUENCE' else path
        resources=[]
        if source=='TILED':
            for number in tiles:
                selected=resource_path.replace('<UDIM>',str(number)).replace('<UVTILE>',f'u{(number-1001)%10+1}_v{(number-1001)//10+1}')
                resources.append((selected,_stat(selected)))
        elif image.use_multiview and image.views_format=='INDIVIDUAL':
            candidates=views or (('left','_L'),('right','_R'))
            stem,extension=os.path.splitext(resource_path)
            previous=next((suffix for _,suffix in candidates if suffix and stem.endswith(suffix)),None)
            index=view if isinstance(view,int) else 0 if view in (None,'','ALL') else next((i for i,value in enumerate(candidates) if value[0]==view),-1)
            selected=(stem[:-len(previous)]+candidates[index][1]+extension) if previous is not None and 0<=index<len(candidates) else resource_path
            resources.append((selected,_stat(selected)))
        else:resources.append((resource_path,_stat(resource_path)))
        metadata=(tuple(resources),bool(image.is_dirty),packed,dimensions)
        color_info=bpy.data.colorspace
        color_context=(color_info.working_space,getattr(color_info,'working_space_interop_id',''))
        return SourceDescriptor((generation,int(image.session_uid)),str(image.name),source,
            image.type,path,image.colorspace_settings.name,color_context,image.alpha_mode,
            str(source_kind).lower(),bool(premultiply),frame,settings,
            layer,pass_name,view,tiles,bool(image.use_multiview),views,stereo_settings,str(channel).upper(),metadata)

    def signature(descriptor,layout=None):
        key=(descriptor.key,None if layout is None else layout.key)
        result=cache.get(key)
        return descriptor.key,0 if result is None else result.revision
    def structure(descriptor):return descriptor.continuity

    def _image(descriptor):
        if descriptor.lifetime[0]!=generation:raise ReferenceError('Scope Image belongs to a retired session')
        image=next((item for item in bpy.data.images if item.session_uid==descriptor.lifetime[1]),None)
        if image is None:raise ReferenceError('The scope Image was removed')
        return image

    def _validate_source(image,descriptor):
        info=bpy.data.colorspace
        color_context=(info.working_space,getattr(info,'working_space_interop_id',''))
        path=bpy.path.abspath(image.filepath,library=image.library) if image.filepath else ''
        current=(image.source,path,image.colorspace_settings.name,image.alpha_mode,
                 bool(image.use_multiview),color_context)
        frozen=(descriptor.source,descriptor.filepath,descriptor.colorspace,descriptor.alpha_mode,
                descriptor.multiview,descriptor.color_context)
        if descriptor.multiview:
            stereo=image.stereo_3d_format
            current+=(image.views_format,stereo.display_mode,stereo.anaglyph_type,
                stereo.interlace_type,stereo.use_interlace_swap,stereo.use_sidebyside_crosseyed,
                stereo.use_squeezed_frame)
            frozen+=descriptor.stereo_settings
        if current!=frozen:
            raise ValueError('The scope Image decoding settings changed after its request was frozen')

    def clear():
        nonlocal generation,sampler_shader
        for (source,_),(handle,_) in pending.items():
            handle.cancel();retiring.setdefault(source,[]).append(handle)
        pending.clear()
        cache.clear();checked.clear();generation+=1;sampler_shader=None

    def invalidate(image=None):
        if image is None:
            checked.clear()
        else:
            uid=int(image.session_uid)
            for key in list(checked):
                if key[0][1][1]==uid:checked.pop(key,None)

    def probe_delay(descriptor):
        now=time.monotonic()
        values=[stamp for key,stamp in checked.items() if key[0]==descriptor.key]
        return max(0.,PROBE_INTERVAL-(now-max(values))) if values else 0.

    def retired(descriptor):
        """Physical receipt for every shared decode of this immutable source.

        Channel cancellation revokes its own interest but must not cancel a
        source helper shared with another consumer. The runtime owns these
        handles until retirement; source cache eviction/Session close may
        request cancellation without weakening that runtime ownership.
        """
        for source,handles in tuple(retiring.items()):
            occupied=[handle for handle in handles if not handle.retired()]
            if occupied:retiring[source]=occupied
            else:retiring.pop(source,None)
        return (descriptor.key not in retiring and
                all(handle.retired() for (source,_),(handle,_) in tuple(pending.items())
                    if source==descriptor.key))

    def _alpha(rgba,image,premultiplied,desired):
        rgba=np.array(rgba,dtype=np.float32,copy=True)
        if image.colorspace_settings.is_data:return rgba
        if image.alpha_mode=='NONE':rgba[...,3]=1.
        if image.alpha_mode in {'NONE','CHANNEL_PACKED'}:return rgba
        if premultiplied and not desired:
            np.divide(rgba[...,:3],rgba[...,3,None],out=rgba[...,:3],where=rgba[...,3,None]!=0)
            rgba[...,:3][rgba[...,3]==0]=0
        elif desired and not premultiplied:rgba[...,:3]*=rgba[...,3,None]
        return rgba

    def _buffer_premultiplied(image):
        if image.type in {'RENDER_RESULT','COMPOSITING','MULTILAYER'}:return True
        if image.type=='UV_TEST':return bool(image.is_float)
        return image.alpha_mode!='STRAIGHT' if image.is_float else image.alpha_mode=='PREMUL'

    def _cpu(image,layout,desired):
        width,height=image.size
        channels=image.channels
        raw=np.empty(width*height*channels,np.float32)
        image.pixels.foreach_get(raw)
        pixels=raw.reshape(height,width,channels)
        if channels in (1,2):
            rgb=np.repeat(pixels[...,:1],3,axis=-1)
            alpha=pixels[...,1:2] if channels==2 else np.ones((height,width,1),np.float32)
            pixels=np.concatenate((rgb,alpha),axis=-1)
        elif channels==3:pixels=np.concatenate((pixels,np.ones((height,width,1),np.float32)),axis=-1)
        if pixels.shape[-1]!=4:raise ValueError('The scope Image has no RGBA pixels')
        rgba=np.array(sample_rgba(pixels,layout),copy=True)
        # Blender stores decoded bytes straight and decoded floats premultiplied.
        # GPU upload may reassociate these to Image.alpha_mode, so its flag is
        # deliberately separate from this CPU buffer contract.
        premultiplied=bool(image.is_float)
        if not image.is_float and not image.colorspace_settings.is_data:
            # Byte buffers retain their encoded source space. OCIO applies the
            # same current Blender configuration as native Image decoding.
            try:import PyOpenColorIO as ocio
            except ImportError:
                # The pip bpy distribution omits the optional Python OCIO
                # binding. Blender's own EXR writer performs the same transform
                # on a private bounded byte buffer, preserving custom configs.
                rgba=_linearize_native(rgba,image)
                premultiplied=image.alpha_mode not in {'NONE','CHANNEL_PACKED'}
            else:
                rgba=_alpha(rgba,image,premultiplied,False)
                rgb=np.ascontiguousarray(rgba[...,:3])
                ocio.GetCurrentConfig().getProcessor(image.colorspace_settings.name,'scene_linear').getDefaultCPUProcessor().applyRGB(rgb)
                rgba[...,:3]=rgb
                premultiplied=False
        return _alpha(rgba,image,premultiplied,desired)

    def _linearize_native(rgba,source):
        height,width=rgba.shape[:2]
        private=bpy.data.images.new('__PG color sample__',width,height,alpha=True,float_buffer=False)
        scene=None
        try:
            private.colorspace_settings.name=source.colorspace_settings.name
            private.alpha_mode=source.alpha_mode
            private.pixels.foreach_set(np.asarray(rgba,dtype=np.float32).ravel())
            scene=bpy.data.scenes.new('__PG color conversion__')
            settings=scene.render.image_settings
            settings.file_format='OPEN_EXR';settings.color_depth='32';settings.exr_codec='ZIPS'
            settings.color_management='OVERRIDE';settings.linear_colorspace_settings.name='scene_linear'
            with tempfile.TemporaryDirectory(prefix='pg-image-color-') as directory:
                path=os.path.join(directory,'sample.exr');private.save_render(path,scene=scene)
                return exr.sample(path,'',pass_name='',size=(width,height))
        finally:
            if scene is not None:bpy.data.scenes.remove(scene)
            bpy.data.images.remove(private)

    def _gpu(image,layout,desired,*,linear=False):
        nonlocal sampler_shader
        import gpu
        from gpu_extras.batch import batch_for_shader
        if bpy.app.background:raise ValueError('Movie scope decoding requires a visible Blender window')
        if sampler_shader is None:
            info=gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0,'VEC2','position')
            info.sampler(0,'FLOAT_2D','inputImage');info.sampler(1,'FLOAT_2D','coordinates')
            info.fragment_out(0,'VEC4','fragColor');info.push_constant('BOOL','interpolatePixels')
            info.vertex_source('void main(){gl_Position=vec4(position,0.0,1.0);}')
            info.fragment_source('vec4 pixel(ivec2 p,ivec2 s){return texelFetch(inputImage,clamp(p,ivec2(0),s-1),0);} void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec2 uv=texelFetch(coordinates,p,0).xy;ivec2 s=textureSize(inputImage,0);if(interpolatePixels){vec2 f=uv*vec2(s)-.5;ivec2 a=ivec2(floor(f));vec2 t=fract(f);fragColor=mix(mix(pixel(a,s),pixel(a+ivec2(1,0),s),t.x),mix(pixel(a+ivec2(0,1),s),pixel(a+ivec2(1,1),s),t.x),t.y);}else{fragColor=pixel(ivec2(uv*vec2(s)),s);}}')
            sampler_shader=gpu.shader.create_from_info(info)
        height,width=layout.shape
        offscreen=gpu.types.GPUOffScreen(width,height,format='RGBA32F')
        coordinates=gpu.types.GPUTexture((width,height),format='RG32F',
            data=gpu.types.Buffer('FLOAT',layout.uv.shape,np.asarray(layout.uv,dtype=np.float32)))
        blend=gpu.state.blend_get();depth=gpu.state.depth_test_get()
        viewport=gpu.state.viewport_get();scissor=gpu.state.scissor_get()
        try:
            with offscreen.bind():
                gpu.state.viewport_set(0,0,width,height);gpu.state.scissor_set(0,0,width,height)
                gpu.state.blend_set('NONE');gpu.state.depth_test_set('NONE')
                shader=sampler_shader;shader.bind();shader.uniform_bool('interpolatePixels',linear)
                shader.uniform_sampler('inputImage',gpu.texture.from_image(image))
                shader.uniform_sampler('coordinates',coordinates)
                batch_for_shader(shader,'TRI_FAN',{'position':[(-1,-1),(1,-1),(1,1),(-1,1)]}).draw(shader)
                rgba=np.array(gpu.state.active_framebuffer_get().read_color(0,0,width,height,4,0,'FLOAT'),dtype=np.float32).reshape(height,width,4)
        finally:
            offscreen.free();gpu.state.viewport_set(*viewport);gpu.state.scissor_set(*scissor)
            gpu.state.blend_set(blend);gpu.state.depth_test_set(depth)
        return _alpha(rgba,image,_buffer_premultiplied(image),desired)

    @contextmanager
    def _loaded(path,source):
        image=bpy.data.images.load(path,check_existing=False)
        try:
            image.colorspace_settings.name=source.colorspace_settings.name
            image.alpha_mode=source.alpha_mode
            yield image
        finally:bpy.data.images.remove(image)

    def _sequence_filepath(path,frame):
        match=list(re.finditer(r'\d+',os.path.splitext(os.path.basename(path))[0]))
        if not match:raise ValueError('The Image sequence filename has no frame number')
        part=match[-1];name=os.path.basename(path)
        return os.path.join(os.path.dirname(path),name[:part.start()]+str(frame).zfill(part.end()-part.start())+name[part.end():])

    def _sequence_path(descriptor):return _sequence_filepath(descriptor.filepath,descriptor.frame)

    def _normalized(image,path):
        """Source decoding only: no scene evaluation or render operator."""
        private=image.copy();scene=None;tree=None
        try:
            tree=bpy.data.node_groups.new('__PG scope source__','CompositorNodeTree')
            tree.nodes.new('CompositorNodeImage').image=private
            scene=bpy.data.scenes.new('__PG scope decode__')
            settings=scene.render.image_settings
            settings.media_type='MULTI_LAYER_IMAGE'
            settings.file_format='OPEN_EXR_MULTILAYER';settings.color_depth='32'
            settings.exr_codec='ZIPS';settings.use_exr_interleave=True
            settings.color_management='OVERRIDE';settings.linear_colorspace_settings.name='scene_linear'
            scene.render.use_multiview=True;settings.views_format='MULTIVIEW'
            private.save_render(path,scene=scene)
        finally:
            if tree is not None:bpy.data.node_groups.remove(tree)
            if scene is not None:bpy.data.scenes.remove(scene)
            bpy.data.images.remove(private)

    def _select_channel(meta,descriptor):
        views=meta['views']
        choice=descriptor.view
        if views and isinstance(choice,int) and not 0<=choice<len(views):
            raise ValueError('The selected Image view is unavailable')
        view=(views[int(choice)] if isinstance(choice,int) else
              (views[0] if choice in (None,'','ALL') else str(choice))) if views else ''
        names=[]
        for name in meta['channels']:
            if not name.endswith('.R'):continue
            parts=name[:-2].split('.')
            parts=[part for part in parts if part not in views]
            if parts:names.append(('.'.join(parts[:-1]),parts[-1]))
        layers=tuple(dict.fromkeys(layer for layer,_ in names))
        choice=descriptor.layer
        if isinstance(choice,int):
            if not layers or not 0<=choice<len(layers):raise ValueError('The selected Image layer is unavailable')
            layer=layers[choice]
        else:layer=str(choice) if choice else (layers[0] if layers else '')
        passes=tuple(sorted({name for candidate,name in names if candidate==layer},key=lambda item:(item!='Combined',item)))
        choice=descriptor.pass_name
        if isinstance(choice,int):
            if not passes or not 0<=choice<len(passes):raise ValueError('The selected Image pass is unavailable')
            pass_name=passes[choice]
        else:pass_name=choice or 'Combined'
        return layer,pass_name,view

    def _multilayer(image,descriptor,layout):
        with tempfile.TemporaryDirectory(prefix='pg-image-source-') as directory:
            path=os.path.join(directory,'source.exr');_normalized(image,path)
            meta=exr.metadata(path);xmin,ymin,xmax,ymax=meta['bounds']
            layout=layout or ordinary_layout(xmax-xmin+1,ymax-ymin+1)
            layer,pass_name,view=_select_channel(meta,descriptor)
            rgba=exr.sample(path,layer,pass_name,view=view,layout=layout)
            rgba=_alpha(rgba,image,True,descriptor.premultiply)
            label=' / '.join(str(part) for part in (layer,pass_name,view) if part)
            return rgba,layout,label

    def _tiles(image,descriptor,layout):
        if descriptor.multiview or descriptor.image_type=='MULTILAYER':
            raise ValueError('Automatic UDIM scopes require ordinary image tiles; per-tile pass or view selection is unavailable')
        if image.is_dirty:raise ValueError('Save the painted UDIM tiles before automatic scope sampling')
        with tempfile.TemporaryDirectory(prefix='pg-image-tiles-') as directory:
            paths={}
            packed={item.tile_number:item for item in image.packed_files if item.view==0}
            for number in descriptor.tile_numbers:
                if number in packed:
                    item=packed[number];suffix=os.path.splitext(item.filepath)[1] or '.exr'
                    path=os.path.join(directory,str(number)+suffix)
                    with open(path,'wb') as stream:stream.write(item.packed_file.data)
                else:
                    path=descriptor.filepath.replace('<UDIM>',str(number))
                    u=(number-1001)%10+1;v=(number-1001)//10+1
                    path=path.replace('<UVTILE>',f'u{u}_v{v}')
                if not os.path.isfile(path):raise ValueError(f'UDIM tile {number} has no stored pixels')
                paths[number]=path
            dimensions=[]
            for number,path in paths.items():
                with _loaded(path,image) as tile:
                    size=tuple(tile.size)
                    if tile.type=='MULTILAYER':
                        raise ValueError('Automatic UDIM scopes require ordinary image tiles; per-tile pass or view selection is unavailable')
                    dimensions.append((number,*size))
            layout=layout or tiled_layout(dimensions)
            rgba=np.zeros((*layout.shape,4),np.float32);incomplete=False
            for number in np.unique(layout.tiles[layout.mass>0]):
                if int(number) not in paths:
                    incomplete=True;continue
                with _loaded(paths[int(number)],image) as tile:
                    sampled=_cpu(tile,layout,descriptor.premultiply)
                mask=layout.tiles==number;rgba[mask]=sampled[mask]
            return rgba,layout,f'{len(paths)} UDIM tiles',incomplete

    def _initialize_views(private,source,scene,directory,filepath=None):
        # The public Image API exposes no ImageUser.scene setter. Image.open
        # sets it on an explicitly supplied private ImageUser, then initializes
        # the source view list. A unique path prevents rebinding another Image;
        # null editor context and undo=False prevent UI/undo side effects.
        extension=os.path.splitext(source.filepath)[1] or '.exr'
        path=os.path.join(directory,'packed-source'+extension)
        if source.packed_files:
            with open(path,'wb') as stream:stream.write(source.packed_files[0].packed_file.data)
        else:shutil.copyfile(filepath or bpy.path.abspath(source.filepath,library=source.library),path)
        private.filepath=path
        tree=bpy.data.node_groups.new('__PG view user__','ShaderNodeTree')
        try:
            node=tree.nodes.new('ShaderNodeTexImage');node.image=private
            with bpy.context.temp_override(area=None,region=None,scene=scene,
                    image_user=node.image_user,texture=None,camera=None):
                result=bpy.ops.image.open('EXEC_DEFAULT',False,filepath=path,
                    use_multiview=True,use_sequence_detection=False,
                    use_udim_detecting=False,relative_path=False)
            if result!={'FINISHED'}:raise ValueError('Blender could not decode the stereo source')
        finally:bpy.data.node_groups.remove(tree)

    def _multiview(image,descriptor,layout):
        views=descriptor.view_settings or (('left','_L'),('right','_R'))
        choice=descriptor.view
        if isinstance(choice,int):
            if not 0<=choice<len(views):raise ValueError('The selected Image view is unavailable')
            index=choice
        else:
            index=0 if choice in (None,'','ALL') else next((i for i,item in enumerate(views) if item[0]==choice),-1)
            if index<0:raise ValueError('The selected Image view is unavailable')
        name,suffix=views[index]
        dirty=bool(image.is_dirty)
        if dirty and descriptor.animated:
            raise ValueError('Save the edited animated multiview Image before automatic frame sampling')
        with tempfile.TemporaryDirectory(prefix='pg-image-views-') as directory:
            packed=next((item for item in image.packed_files if item.view==index),None)
            path=None
            if image.views_format=='INDIVIDUAL' and not dirty:
                if packed is not None:
                    path=os.path.join(directory,'view'+(os.path.splitext(packed.filepath)[1] or '.exr'))
                    with open(path,'wb') as stream:stream.write(packed.packed_file.data)
                else:
                    stem,extension=os.path.splitext(descriptor.filepath)
                    previous=next((value for _,value in views if value and stem.endswith(value)),None)
                    if previous is not None:path=stem[:-len(previous)]+suffix+extension
                    elif index==0:path=descriptor.filepath
                if path is None or not os.path.isfile(path):raise ValueError('The selected Image view has no stored pixels: '+name)
                with _loaded(path,image) as private:
                    layout=layout or ordinary_layout(*private.size)
                    rgba=_cpu(private,layout,descriptor.premultiply)
                return rgba,layout,name,False
            # The native writer can read every initialized resident view. Use
            # save_render's save-copy semantics for painted images: copying an
            # Image would lose its live per-view buffers. Clean sources stay
            # entirely private, including their source view initialization.
            private=image if dirty else image.copy();scene=None
            try:
                scene=bpy.data.scenes.new('__PG view decode__')
                scene.render.use_multiview=True
                scene.render.views_format='MULTIVIEW'
                for item in scene.render.views:item.use=False
                for view_name,view_suffix in views:
                    item=scene.render.views.get(view_name) or scene.render.views.new(view_name)
                    item.file_suffix=view_suffix;item.use=True
                settings=scene.render.image_settings
                settings.file_format='OPEN_EXR';settings.color_depth='32';settings.exr_codec='ZIPS'
                settings.views_format='INDIVIDUAL';settings.color_management='OVERRIDE'
                settings.linear_colorspace_settings.name='scene_linear'
                if not dirty:
                    if private.source=='SEQUENCE':private.source='FILE'
                    _initialize_views(private,image,scene,directory,descriptor.filepath)
                path=os.path.join(directory,'view.exr')
                selected=os.path.join(directory,'view'+suffix+'.exr')
                try:private.save_render(path,scene=scene)
                except RuntimeError:
                    # Blender 5.2's individual-view writer reports failure after
                    # successful writes (its success accumulator starts false).
                    # Accept only a complete, subsequently validated EXR file.
                    if not os.path.isfile(selected):raise
                if not os.path.isfile(selected):
                    raise ValueError('The selected Image view has no initialized resident pixels; open the source in Blender before automatic scope sampling')
                meta=exr.metadata(selected);xmin,ymin,xmax,ymax=meta['bounds']
                layout=layout or ordinary_layout(xmax-xmin+1,ymax-ymin+1)
                rgba=exr.sample(selected,'',pass_name='',layout=layout)
                source_is_float=True
                if image.views_format=='STEREO_3D' and not image.colorspace_settings.is_data:
                    # Never query a live static multiview buffer with the RNA
                    # null-ImageUser getter: it can evict its resident eyes.
                    metadata_image=image.copy() if dirty else private
                    try:source_is_float=bool(metadata_image.is_float)
                    finally:
                        if dirty:bpy.data.images.remove(metadata_image)
                if not source_is_float:
                    # Blender's stereo splitter drops byte-buffer colorspace
                    # metadata. Its EXR writer therefore preserves encoded RGB
                    # (premultiplied); apply the configured source transform here.
                    rgba=_alpha(rgba,image,True,False)
                    try:import PyOpenColorIO as ocio
                    except ImportError:rgba=_linearize_native(rgba,image)
                    else:
                        rgb=np.ascontiguousarray(rgba[...,:3])
                        ocio.GetCurrentConfig().getProcessor(image.colorspace_settings.name,'scene_linear').getDefaultCPUProcessor().applyRGB(rgb)
                        rgba[...,:3]=rgb
                        rgba=_alpha(rgba,image,False,True)
                return _alpha(rgba,image,True,descriptor.premultiply),layout,name,False
            finally:
                if scene is not None:bpy.data.scenes.remove(scene)
                if not dirty:bpy.data.images.remove(private)

    def _movie(image,descriptor,layout):
        if bpy.app.background:
            if background_movies:
                from .image_decode import movie
                return movie(image,descriptor,layout,exr,_alpha,_linearize_native)
            raise ValueError('Movie scope decoding requires a visible Blender window')
        # Discovery leaves native duration metadata cold. Its foreground
        # fallback is guarded; the helper checks the selected private view.
        descriptor=replace(descriptor,frame=max(1,min(descriptor.frame,max(1,int(image.frame_duration)))))
        private=image.copy()
        try:
            detail=f'frame {descriptor.frame}'
            if not descriptor.multiview:
                private.gl_free();private.gl_load(frame=max(1,descriptor.frame))
                layout=layout or ordinary_layout(*private.size)
                return _gpu(private,layout,descriptor.premultiply),layout,detail,False
            views=descriptor.view_settings or (('left','_L'),('right','_R'))
            choice=descriptor.view
            index=(choice if isinstance(choice,int) else 0 if choice in (None,'','ALL') else
                   next((i for i,item in enumerate(views) if item[0]==choice),-1))
            if not 0<=index<len(views):raise ValueError('The selected movie view is unavailable')
            view,suffix=views[index];detail+=' / '+view
            private.use_multiview=False
            if image.views_format=='INDIVIDUAL':
                stem,extension=os.path.splitext(descriptor.filepath)
                previous=next((value for _,value in views if value and stem.endswith(value)),None)
                if previous is None and index!=0:raise ValueError('The selected movie view has no stored pixels')
                if previous is not None:private.filepath=stem[:-len(previous)]+suffix+extension
                if not os.path.isfile(bpy.path.abspath(private.filepath)):raise ValueError('The selected movie view has no stored pixels')
                private.gl_free();private.gl_load(frame=max(1,descriptor.frame))
                layout=layout or ordinary_layout(*private.size)
                return _gpu(private,layout,descriptor.premultiply),layout,detail,False
            if index>1:raise ValueError('Packed stereo movies contain only left and right views')
            # Decode the packed frame once. GPU sampling gathers only the
            # requested eye's bounded pixel centers; missing interlace pixels
            # and anaglyph channels follow Blender's stereo decoder convention.
            private.colorspace_settings.name='Non-Color'
            private.gl_free();private.gl_load(frame=max(1,descriptor.frame))
            width,height=private.size;is_float=bool(private.is_float)
            settings=image.stereo_3d_format;mode=settings.display_mode
            eye_width=width if settings.use_squeezed_frame or mode!='SIDEBYSIDE' else width//2
            eye_height=height if settings.use_squeezed_frame or mode!='TOPBOTTOM' else height//2
            layout=layout or ordinary_layout(eye_width,eye_height)
            uv=np.array(layout.uv,copy=True)
            linear=False
            if mode=='SIDEBYSIDE':
                eye=index ^ bool(settings.use_sidebyside_crosseyed)
                uv[...,0]=(uv[...,0]+eye)*.5;linear=settings.use_squeezed_frame
            elif mode=='TOPBOTTOM':
                uv[...,1]=(uv[...,1]+1-index)*.5;linear=settings.use_squeezed_frame
            mapped=replace(layout,uv=immutable(uv))
            rgba=_gpu(private,mapped,False,linear=linear)
            if linear and not is_float:rgba=np.floor(np.clip(rgba,0.,1.)*255.+.5)/255.
            if mode=='INTERLACE':
                x=np.floor(layout.uv[...,0]*eye_width).astype(int)
                y=np.floor(layout.uv[...,1]*eye_height).astype(int)
                parity=y if settings.interlace_type=='ROW_INTERLEAVED' else x if settings.interlace_type=='COLUMN_INTERLEAVED' else x+y
                eye=(parity+int(settings.use_interlace_swap)+int(not is_float))%2
                rgba[eye!=index]=0
            elif mode=='ANAGLYPH':
                owners={'RED_CYAN':(0,1,1),'GREEN_MAGENTA':(1,0,1),'YELLOW_BLUE':(0,0,1)}[settings.anaglyph_type]
                for channel,owner in enumerate(owners):
                    if owner!=index:rgba[...,channel]=0
            if not image.colorspace_settings.is_data:
                rgba=_alpha(rgba,image,image.alpha_mode=='PREMUL',False)
                try:import PyOpenColorIO as ocio
                except ImportError:rgba=_linearize_native(rgba,image);premultiplied=True
                else:
                    rgb=np.ascontiguousarray(rgba[...,:3])
                    ocio.GetCurrentConfig().getProcessor(image.colorspace_settings.name,'scene_linear').getDefaultCPUProcessor().applyRGB(rgb)
                    rgba[...,:3]=rgb;premultiplied=False
                rgba=_alpha(rgba,image,premultiplied,descriptor.premultiply)
            return rgba,layout,detail,False
        finally:bpy.data.images.remove(private)

    def _decode(image,descriptor,layout):
        if descriptor.source=='TILED':return _tiles(image,descriptor,layout)
        if descriptor.source=='SEQUENCE':
            path=_sequence_path(descriptor)
            if not os.path.isfile(path):raise ValueError(f'Image sequence frame {descriptor.frame} is missing')
            if descriptor.multiview:
                selected=replace(descriptor,filepath=path)
                rgba,layout,detail,incomplete=_multiview(image,selected,layout)
                return rgba,layout,f'frame {descriptor.frame} / '+detail,incomplete
            with _loaded(path,image) as private:
                dimensions=tuple(private.size)
                if private.type=='MULTILAYER':
                    rgba,layout,label=_multilayer(private,descriptor,layout)
                else:
                    layout=layout or ordinary_layout(*dimensions)
                    rgba=_cpu(private,layout,descriptor.premultiply);label=''
            return rgba,layout,f'frame {descriptor.frame}'+(' / '+label if label else ''),False
        if descriptor.source=='MOVIE':return _movie(image,descriptor,layout)
        dimensions=None if descriptor.multiview else tuple(image.size)
        if image.type=='MULTILAYER':
            rgba,layout,label=_multilayer(image,descriptor,layout)
            return rgba,layout,label,False
        if descriptor.multiview:return _multiview(image,descriptor,layout)
        layout=layout or ordinary_layout(*dimensions)
        return _cpu(image,layout,descriptor.premultiply),layout,'',False

    def _stored_request(image,descriptor,layout):
        """Copy encoded storage; never acquire a foreground Image buffer."""
        from dataclasses import asdict
        packed=[]
        views=descriptor.view_settings or (('left','_L'),('right','_R'))
        for item in image.packed_files:
            path=descriptor.filepath or bpy.path.abspath(item.filepath)
            if descriptor.source=='TILED':
                number=int(item.tile_number)
                path=path.replace('<UDIM>',str(number)).replace('<UVTILE>',
                    f'u{(number-1001)%10+1}_v{(number-1001)//10+1}')
            elif descriptor.multiview and descriptor.stereo_settings[0]=='INDIVIDUAL':
                stem,extension=os.path.splitext(path)
                suffix=next((suffix for _,suffix in views if suffix and stem.endswith(suffix)),None)
                if suffix is not None and 0<=item.view<len(views):
                    path=stem[:-len(suffix)]+views[item.view][1]+extension
            packed.append(dict(path=path,tile=int(item.tile_number),view=int(item.view),
                data=np.frombuffer(item.packed_file.data,dtype=np.uint8).copy()))
        return dict(descriptor=asdict(descriptor),key=descriptor.key,packed=tuple(packed),
            layout=None if layout is None else {name:getattr(layout,name)
                for name in ('mass','x','uv','tiles','ids','key','tiled')})

    def _stored_decode(descriptor,layout,key):
        from ...blender_runtime.execution.contracts import JobSpec
        attempt=pending.get(key)
        if attempt is None:
            image=_image(descriptor);_validate_source(image,descriptor)
            if image.is_dirty:raise SourcePreparing('Scope Image needs a safe resident snapshot')
            request=_stored_request(image,descriptor,layout)
            runtime=_application.component('services').get_runtime()
            handle=runtime.submit(JobSpec('scope_image_read',request,lane='capture',key=None))
            packed_digest=tuple(hashlib.sha256(item['data']).digest() for item in request['packed'])
            attempt=pending[key]=(handle,packed_digest)
            # The execution owner retains cancelled physical work until its
            # retirement receipt. This cache holds no RNA or native resources.
            while len(pending)>16:
                (source,_),(expired,_)=pending.popitem(last=False)
                expired.cancel();retiring.setdefault(source,[]).append(expired)
        handle,packed_digest=attempt
        if not handle.retired():raise SourcePreparing('Preparing source Image samples')
        pending.pop(key,None)
        result=handle.result()
        if result['key']!=descriptor.key:raise ValueError('Source Image receipt belongs to another request')
        image=_image(descriptor);_validate_source(image,descriptor)
        current_digest=tuple(hashlib.sha256(item.packed_file.data).digest() for item in image.packed_files)
        if current_digest!=packed_digest or bool(image.is_dirty)!=descriptor.metadata[1]:
            raise ValueError('Source Image pixels changed after its request was frozen')
        if any(_stat(path)!=stamp for path,stamp in descriptor.metadata[0]):
            raise ValueError('Source Image storage changed after its request was frozen')
        from ...perceptual_grade_core.scopes.samples import SampleLayout
        values=result['layout']
        actual=SampleLayout(**{name:immutable(values[name]) if name in ('mass','x','uv','tiles','ids') else values[name]
            for name in ('mass','x','uv','tiles','ids','key','tiled')})
        if layout is not None and actual.key!=layout.key:
            raise ValueError('Source Image receipt changed its requested sample grid')
        return result['rgba'],actual,result['detail'],result['incomplete']

    def read(descriptor,*,layout=None,force=False,asynchronous=False):
        nonlocal sequence
        if diagnostic_state['enabled']:diagnostic_state['reads']+=1
        key=(descriptor.key,None if layout is None else layout.key)
        now=time.monotonic();previous=cache.get(key)
        if previous is not None and not force and now-checked.get(key,-float('inf'))<PROBE_INTERVAL:
            cache.move_to_end(key);return previous
        stored=(descriptor.source in {'FILE','SEQUENCE','MOVIE','TILED'} and not descriptor.metadata[1])
        if asynchronous and not stored and previous is not None and not guard.available():
            # Retain the accepted immutable snapshot while a native reader is
            # active. Leave its probe timestamp stale so the first safe tick
            # checks the resident buffer again.
            _validate_source(_image(descriptor),descriptor)
            cache.move_to_end(key);return previous
        if asynchronous and stored:
            rgba,actual,detail,incomplete=_stored_decode(descriptor,layout,key)
        else:
            with guard.native_operation():
                image=_image(descriptor)
                _validate_source(image,descriptor)
                started=time.perf_counter() if diagnostic_state['enabled'] else None
                try:rgba,actual,detail,incomplete=_decode(image,descriptor,layout)
                finally:
                    if started is not None:
                        elapsed=time.perf_counter()-started
                        diagnostic_state['decodes']+=1;diagnostic_state['decode_seconds']+=elapsed
                        diagnostic_state['events'].append((started,elapsed,descriptor.key,layout.key if layout is not None else None))
                        del diagnostic_state['events'][:-64]
        rgba=np.array(rgba,copy=True)
        if descriptor.channel=='ALPHA':
            rgba=np.concatenate((np.repeat(rgba[...,3,None],3,axis=-1),np.ones((*rgba.shape[:2],1),np.float32)),axis=-1)
        elif descriptor.channel!='COLOR':raise ValueError('Unsupported Image output '+descriptor.channel)
        rgba[actual.mass==0]=0
        label=descriptor.name+(' · '+detail if detail else '')
        if incomplete:label+=' · incomplete reference tiles'
        checked[key]=now
        if previous is not None and previous.layout.key==actual.key and previous.label==label and np.array_equal(previous.rgba,rgba,equal_nan=True):
            cache.move_to_end(key);return previous
        sequence+=1
        result=SampleResult(immutable(rgba),actual,key,sequence,label,incomplete)
        cache[key]=result;cache.move_to_end(key)
        while len(cache)>32:
            expired,_=cache.popitem(last=False);checked.pop(expired,None)
        return result

    def _set_time(value):
        nonlocal time
        time=value

    return Component({'describe':describe,'read':read,'clear':clear,'invalidate':invalidate,
                      'signature':signature,'structure':structure,'probe_delay':probe_delay,
                      'retired':retired,
                      'enable_diagnostics':enable_diagnostics,'diagnostics':diagnostics,
                      'FrameSettings':FrameSettings,'SourceDescriptor':SourceDescriptor,
                      'SampleResult':SampleResult,'SourcePreparing':SourcePreparing,'PROBE_INTERVAL':PROBE_INTERVAL},
                     readers={'time':lambda:time},writers={'time':_set_time})
