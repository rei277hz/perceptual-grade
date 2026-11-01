"""Decode immutable stored Image inputs in the isolated capture process.

The helper enables no extension Session and never receives foreground RNA.
Encoded packed bytes are sealed at submission; disk resources must still match
the metadata frozen with the request before and after their private copy.
"""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np


def _stat(path):
    value=path.stat()
    return value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _resources(inputs,directory,context):
    descriptor=inputs['descriptor'];copied={}
    packed={item['path']:item for item in inputs['packed']}
    for path,stamp in descriptor['metadata'][0]:
        target=directory/Path(path).name
        if path in packed:
            target.write_bytes(np.asarray(packed[path]['data'],dtype=np.uint8).tobytes())
        else:
            source=Path(path)
            if stamp is None or not source.is_file():raise ValueError('Source Image storage is missing: '+path)
            if _stat(source)!=tuple(stamp):raise ValueError('Source Image storage changed after its request was frozen')
            with source.open('rb') as origin,target.open('wb') as destination:
                while block:=origin.read(1024*1024):
                    context.checkpoint();destination.write(block)
            if _stat(source)!=tuple(stamp):raise ValueError('Source Image storage changed during its private snapshot')
        copied[path]=str(target)
    # Packed view/tile bytes may be the only remaining copy of the source.
    for path,item in packed.items():
        if path not in copied:
            target=directory/Path(path).name
            target.write_bytes(np.asarray(item['data'],dtype=np.uint8).tobytes())
            copied[path]=str(target)
    context.checkpoint()
    return copied


def handle(inputs,context):
    import bpy
    from ...scopes import images,exr
    from ....perceptual_grade_core.scopes.samples import SampleLayout
    from ...integration.components import Component
    class PrivateApplication:
        def component(self,name):
            if name=='graph.output_guard':return Component({'native_operation':nullcontext})
            if name=='scopes.exr':return exr.create(None)
            raise ValueError('Source decoder requested an application service: '+name)
    value=dict(inputs['descriptor'])
    value['frame_settings']=images.FrameSettings(**value['frame_settings'])
    descriptor=images.SourceDescriptor(**value)
    original_colorspace=bpy.data.colorspace.working_space
    original_images=set(bpy.data.images)
    original_scenes=set(bpy.data.scenes)
    original_groups=set(bpy.data.node_groups)
    try:
        if original_colorspace!=descriptor.color_context[0]:
            bpy.ops.wm.set_working_color_space(working_space=descriptor.color_context[0])
        with tempfile.TemporaryDirectory(prefix='pg-stored-image-') as temporary:
            directory=Path(temporary);resources=_resources(inputs,directory,context)
            first=next(iter(resources.values()),None)
            if first is None:raise ValueError('Source Image has no stored pixels')
            image=bpy.data.images.load(first,check_existing=False)
            image.source=descriptor.source
            image.filepath=str(directory/(Path(descriptor.filepath).name or Path(first).name))
            image.colorspace_settings.name=descriptor.colorspace;image.alpha_mode=descriptor.alpha_mode
            image.use_multiview=descriptor.multiview
            if descriptor.multiview:
                fields=('display_mode','anaglyph_type','interlace_type','use_interlace_swap',
                        'use_sidebyside_crosseyed','use_squeezed_frame')
                image.views_format=descriptor.stereo_settings[0]
                for name,setting in zip(fields,descriptor.stereo_settings[1:]):setattr(image.stereo_3d_format,name,setting)
            for number in descriptor.tile_numbers:
                if number not in {tile.number for tile in image.tiles}:image.tiles.new(number)
            local=replace(descriptor,lifetime=(0,int(image.session_uid)),filepath=image.filepath,channel='COLOR')
            reader=images.create(PrivateApplication(),background_movies=True)
            layout=None if inputs['layout'] is None else SampleLayout(**dict(inputs['layout']))
            result=reader.read(local,layout=layout,force=True)
            context.checkpoint()
            label=result.label
            if label.startswith(descriptor.name+' · '):label=label[len(descriptor.name)+3:]
            elif label==descriptor.name:label=''
            if result.incomplete:label=label.removesuffix(' · incomplete reference tiles')
            return dict(key=inputs['key'],rgba=result.rgba,detail=label,incomplete=result.incomplete,
                layout={name:getattr(result.layout,name) for name in ('mass','x','uv','tiles','ids','key','tiled')})
    finally:
        for scene in set(bpy.data.scenes)-original_scenes:bpy.data.scenes.remove(scene)
        for tree in set(bpy.data.node_groups)-original_groups:bpy.data.node_groups.remove(tree)
        for image in set(bpy.data.images)-original_images:bpy.data.images.remove(image)
        if bpy.data.colorspace.working_space!=original_colorspace:
            bpy.ops.wm.set_working_color_space(working_space=original_colorspace)
