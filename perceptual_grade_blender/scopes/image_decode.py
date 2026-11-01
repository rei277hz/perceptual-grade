"""Source-only background movie decoding, independent of product lifecycle."""
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
import tempfile
import numpy as np


@contextmanager
def _frame(image,descriptor,exr):
    """Blender evaluates only one private Image node at the frozen movie frame."""
    import bpy
    scene=tree=camera=object_=None
    try:
        scene=bpy.data.scenes.new('__PG movie source__')
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1
        scene.render.resolution_x=scene.render.resolution_y=1;scene.render.resolution_percentage=100
        scene.render.use_compositing=True;scene.render.use_sequencer=False
        camera=bpy.data.cameras.new('__PG movie camera__');object_=bpy.data.objects.new(camera.name,camera)
        scene.collection.objects.link(object_);scene.camera=object_
        tree=bpy.data.node_groups.new('__PG movie source__','CompositorNodeTree')
        scene.compositing_node_group=tree
        source=tree.nodes.new('CompositorNodeImage');source.image=image
        source.frame_start=1;source.frame_duration=max(descriptor.frame,int(image.frame_duration),1)
        source.use_auto_refresh=True
        tree.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor')
        endpoint=tree.nodes.new('NodeGroupOutput');tree.links.new(source.outputs['Image'],endpoint.inputs['Image'])
        output=tree.nodes.new('CompositorNodeOutputFile');output.file_name='source_'
        item=output.file_output_items.new('RGBA','pixels')
        settings=output.format;settings.media_type='IMAGE';settings.file_format='OPEN_EXR'
        settings.color_mode='RGBA';settings.color_depth='32';settings.exr_codec='ZIPS'
        settings.color_management='OVERRIDE';settings.linear_colorspace_settings.name='scene_linear'
        output.save_as_render=False;tree.links.new(source.outputs['Image'],output.inputs[item.name])
        with tempfile.TemporaryDirectory(prefix='pg-movie-frame-') as directory:
            output.directory=directory;scene.frame_set(max(1,descriptor.frame))
            with bpy.context.temp_override(scene=scene):bpy.ops.render.render(write_still=False,scene=scene.name)
            path=str(Path(directory)/'source_pixels.exr')
            metadata=exr.metadata(path);xmin,ymin,xmax,ymax=metadata['bounds']
            width=xmax-xmin+1;height=ymax-ymin+1
            yield path,width,height
    finally:
        if scene is not None:bpy.data.scenes.remove(scene)
        if tree is not None:bpy.data.node_groups.remove(tree)
        if object_ is not None:bpy.data.objects.remove(object_)
        if camera is not None:bpy.data.cameras.remove(camera)


def _sample(exr,path,width,height,layout,uv,linear=False):
    def at(x,y):
        coordinates=np.stack(((np.clip(x,0,width-1)+.5)/width,(np.clip(y,0,height-1)+.5)/height),axis=-1)
        return exr.sample(path,'',pass_name='',layout=replace(layout,uv=coordinates))
    if not linear:return at(np.floor(uv[...,0]*width).astype(int),np.floor(uv[...,1]*height).astype(int))
    x=uv[...,0]*width-.5;y=uv[...,1]*height-.5
    a=np.floor(x).astype(int);b=np.floor(y).astype(int);tx=(x-a)[...,None];ty=(y-b)[...,None]
    return (at(a,b)*(1-tx)+at(a+1,b)*tx)*(1-ty)+(at(a,b+1)*(1-tx)+at(a+1,b+1)*tx)*ty


def movie(image,descriptor,layout,exr,alpha,linearize):
    import bpy
    from ...perceptual_grade_core.scopes.samples import ordinary_layout
    private=image.copy()
    try:
        detail=f'frame {descriptor.frame}'
        index=0;packed=False
        if descriptor.multiview:
            views=descriptor.view_settings or (('left','_L'),('right','_R'))
            choice=descriptor.view
            index=choice if isinstance(choice,int) else 0 if choice in (None,'','ALL') else next((i for i,item in enumerate(views) if item[0]==choice),-1)
            if not 0<=index<len(views):raise ValueError('The selected movie view is unavailable')
            view,suffix=views[index];detail+=' / '+view
            private.use_multiview=False
            if image.views_format=='INDIVIDUAL':
                path=Path(descriptor.filepath);stem=path.stem
                previous=next((suffix for _,suffix in views if suffix and stem.endswith(suffix)),None)
                if previous is None and index!=0:raise ValueError('The selected movie view has no stored pixels')
                if previous is not None:private.filepath=str(path.with_name(stem[:-len(previous)]+suffix+path.suffix))
                if not Path(private.filepath).is_file():raise ValueError('The selected movie view has no stored pixels')
            else:
                if index>1:raise ValueError('Packed stereo movies contain only left and right views')
                packed=True;private.colorspace_settings.name='Non-Color'
        descriptor=replace(descriptor,frame=max(1,min(descriptor.frame,max(1,int(private.frame_duration)))))
        detail=f'frame {descriptor.frame}'+(' / '+view if descriptor.multiview else '')
        with _frame(private,descriptor,exr) as (path,width,height):
            if not packed:
                layout=layout or ordinary_layout(width,height)
                return alpha(_sample(exr,path,width,height,layout,layout.uv),image,True,descriptor.premultiply),layout,detail,False
            settings=image.stereo_3d_format;mode=settings.display_mode
            is_float=bool(private.is_float)
            eye_width=width if settings.use_squeezed_frame or mode!='SIDEBYSIDE' else width//2
            eye_height=height if settings.use_squeezed_frame or mode!='TOPBOTTOM' else height//2
            layout=layout or ordinary_layout(eye_width,eye_height);uv=np.array(layout.uv,copy=True);linear=False
            if mode=='SIDEBYSIDE':
                uv[...,0]=(uv[...,0]+(index ^ bool(settings.use_sidebyside_crosseyed)))*.5;linear=settings.use_squeezed_frame
            elif mode=='TOPBOTTOM':uv[...,1]=(uv[...,1]+1-index)*.5;linear=settings.use_squeezed_frame
            rgba=_sample(exr,path,width,height,layout,uv,linear)
            # Native stereo splitting rounds only decoded byte buffers. Videos
            # above eight bits retain floating precision and opposite interlace
            # ownership, exactly as the visible Image decoder does.
            if linear and not is_float:rgba=np.floor(np.clip(rgba,0.,1.)*255.+.5)/255.
            if mode=='INTERLACE':
                x=np.floor(layout.uv[...,0]*eye_width).astype(int);y=np.floor(layout.uv[...,1]*eye_height).astype(int)
                parity=y if settings.interlace_type=='ROW_INTERLEAVED' else x if settings.interlace_type=='COLUMN_INTERLEAVED' else x+y
                rgba[(parity+int(settings.use_interlace_swap)+int(not is_float))%2!=index]=0
            elif mode=='ANAGLYPH':
                owners={'RED_CYAN':(0,1,1),'GREEN_MAGENTA':(1,0,1),'YELLOW_BLUE':(0,0,1)}[settings.anaglyph_type]
                for channel,owner in enumerate(owners):
                    if owner!=index:rgba[...,channel]=0
            if not image.colorspace_settings.is_data:
                rgba=alpha(rgba,image,image.alpha_mode=='PREMUL',False)
                try:import PyOpenColorIO as ocio
                except ImportError:rgba=linearize(rgba,image);premultiplied=True
                else:
                    rgb=np.ascontiguousarray(rgba[...,:3]);ocio.GetCurrentConfig().getProcessor(image.colorspace_settings.name,'scene_linear').getDefaultCPUProcessor().applyRGB(rgb)
                    rgba[...,:3]=rgb;premultiplied=False
                rgba=alpha(rgba,image,premultiplied,descriptor.premultiply)
            return rgba,layout,detail,False
    finally:bpy.data.images.remove(private)
