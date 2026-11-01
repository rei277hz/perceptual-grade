"""Read a small pixel grid from Blender's scanline render-pass snapshot.

Only the format we write is accepted: single-part, 32-bit float channels,
uncompressed or one-row ZIP (ZIPS). No external EXR module is required.
Coordinates remain bottom-up, matching Blender Image.pixels and our scopes.
"""

from ..integration.components import Component

def create(_application):
    import struct
    import zlib
    import numpy as np



    def _read(stream,size):
        data=stream.read(size)
        if len(data)!=size:raise ValueError('Incomplete compositor scope snapshot')
        return data


    def _string(stream):
        result=bytearray()
        while True:
            char=_read(stream,1)
            if char==b'\0':return result.decode('utf8')
            result+=char
            if len(result)>1024:raise ValueError('Invalid compositor scope snapshot name')


    def _metadata(stream):
        magic,version=struct.unpack('<II',_read(stream,8))
        if magic!=20000630 or version&0x1a00:raise ValueError('Unsupported compositor scope snapshot layout')
        header={}
        while True:
            name=_string(stream)
            if not name:break
            kind=_string(stream);length=struct.unpack('<I',_read(stream,4))[0]
            if length>16*1024*1024:raise ValueError('Invalid compositor scope snapshot header')
            header[name]=(kind,_read(stream,length))
        if header['compression'][1] not in (b'\0',b'\2'):
            raise ValueError('Compositor scope snapshot requires uncompressed or ZIPS data')
        import io
        ch=io.BytesIO(header['channels'][1]);channels=[]
        while True:
            name=_string(ch)
            if not name:break
            pixel_type,_,xs,ys=struct.unpack('<iB3xii',_read(ch,16))
            if pixel_type!=2 or xs!=1 or ys!=1:raise ValueError('Compositor scope snapshot requires full-resolution float channels')
            channels.append(name)
        bounds=struct.unpack('<4i',header['dataWindow'][1])
        views=[]
        if 'multiView' in header:
            values=io.BytesIO(header['multiView'][1])
            while values.tell()<len(header['multiView'][1]):
                length=struct.unpack('<I',_read(values,4))[0]
                views.append(_read(values,length).decode('utf8'))
        return dict(channels=tuple(channels),bounds=bounds,
                    compression=header['compression'][1][0],views=tuple(views))


    def _header(stream):
        meta=_metadata(stream)
        return meta['channels'],meta['bounds'],meta['compression']


    def metadata(path):
        with open(path,'rb') as stream:return _metadata(stream)



    def _unzip(data,length):
        if len(data)==length:return data
        encoded=np.frombuffer(zlib.decompress(data),dtype=np.uint8)
        if len(encoded)!=length:raise ValueError('Invalid compositor scope snapshot row')
        predicted=encoded.astype(np.int64);predicted[1:]-=128
        restored=(np.cumsum(predicted)&255).astype(np.uint8)
        raw=np.empty(length,np.uint8);middle=(length+1)//2
        raw[::2]=restored[:middle];raw[1::2]=restored[middle:]
        return raw.tobytes()


    def sample(path,layer='',pass_name='Combined',size=None,view='',layout=None):
        with open(path,'rb') as stream:
            channels,(xmin,ymin,xmax,ymax),compression=_header(stream)
            width,height=xmax-xmin+1,ymax-ymin+1
            if min(width,height)<=0:raise ValueError('Empty compositor scope snapshot')
            root=(str(layer)+'.') if layer else ''
            prefixes=[root+str(pass_name)+'.'] if pass_name else [root]
            if view:
                prefixes=[root+str(pass_name)+'.'+view+'.',root+view+'.'+str(pass_name)+'.'] if pass_name else [root+view+'.']
            if not layer and pass_name=='Combined':
                prefixes += [view+'.'] if view else ['']
            prefix=next((p for p in prefixes if all(p+c in channels for c in 'RGB')),None)
            if prefix is None:raise ValueError('Render pass {layer} / {pass_name} / {view} has no RGB pixels'.format(layer=layer,pass_name=pass_name,view=view))
            ids=[channels.index(prefix+c) for c in 'RGB'];alpha=channels.index(prefix+'A') if prefix+'A' in channels else None
            if layout is None:
                sw,sh=size or (min(256,width),min(192,height))
                xs,ys=np.meshgrid(np.minimum(width-1,((np.arange(sw)+.5)*width/sw).astype(int)),
                                 np.minimum(height-1,((np.arange(sh)+.5)*height/sh).astype(int)))
            else:
                from ...perceptual_grade_core.scopes.samples import pixel_indices
                xs,ys=pixel_indices(layout,width,height)
            result=np.ones((*xs.shape,4),np.float32)
            offsets=struct.unpack('<'+'Q'*height,_read(stream,height*8))
            length=width*len(channels)*4
            # File scanlines are top-down, scope coordinates bottom-up.
            for index in np.unique(height-1-ys):
                stream.seek(offsets[int(index)]);y,packed=struct.unpack('<iI',_read(stream,8))
                if y!=ymin+index:raise ValueError('Unexpected compositor scope snapshot scanline')
                data=_read(stream,packed)
                if compression:data=_unzip(data,length)
                if len(data)!=length:raise ValueError('Invalid compositor scope snapshot pixel count')
                row=np.frombuffer(data,dtype='<f4').reshape(len(channels),width)
                mask=height-1-ys==index
                result[mask,:3]=row[ids][:,xs[mask]].T
                if alpha is not None:result[mask,3]=row[alpha,xs[mask]]
            if layout is not None:result[layout.mass==0]=0
            return result

    def _set_export__read(value):
        nonlocal _read
        _read = value

    def _set_export__string(value):
        nonlocal _string
        _string = value

    def _set_export__header(value):
        nonlocal _header
        _header = value

    def _set_export__unzip(value):
        nonlocal _unzip
        _unzip = value

    def _set_export_sample(value):
        nonlocal sample
        sample = value

    return Component(readers={
        "__name__": lambda: __name__,
        'metadata': lambda: metadata,
        '_header': lambda: _header,
        '_read': lambda: _read,
        '_string': lambda: _string,
        '_unzip': lambda: _unzip,
        'np': lambda: np,
        'sample': lambda: sample,
        'struct': lambda: struct,
        'zlib': lambda: zlib,
    }, writers={
    '_read': _set_export__read, '_string': _set_export__string, '_header': _set_export__header, '_unzip': _set_export__unzip, 'sample': _set_export_sample})
