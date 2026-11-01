"""33/65-cube lookup, coalesced previews, cancellable refinement and assets."""

from ..integration.components import Component

def create(_application):
    from ..runtime.protocol import Protocol
    from itertools import count
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text
    remove_image = _application.app.blender.native.remove_image
    tr = _application.component('localization.i18n').tr; error_text = _application.component('localization.i18n').error_text
    from ...blender_runtime.execution.diagnostics import log_exception

    from collections import OrderedDict
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    from concurrent.futures import CancelledError
    import logging
    import json
    import math
    import time
    import bpy
    import numpy as np
    from ...perceptual_grade_core.model import lut as cm
    scheduler = _application.component('graph.scheduler')
    resources = _application.component('graph.output_resources')

    SCHEMA=25
    LOG=logging.getLogger(__name__)
    ROUTE='PG Evaluation Route'
    _states=OrderedDict()
    _cubes={}
    _ready=OrderedDict()
    _scope_demands=OrderedDict()
    _scope_running=None
    _scope_failures={}
    _failures={}
    _retired=set()
    _retirements={}
    _publication_serial=count(1)
    PREVIEW_DELAY=.04


    class ScopePreparing(RuntimeError):
        """A final numerical cube is being prepared for a scope snapshot."""


    class ScopeFailure(RuntimeError):
        """The demanded final numerical cube failed to bake."""


    class _Graph:
        def __init__(self,tree,source):
            self.tree=tree;self.source=source;self.cache={};self.index=0

        def op(self,operation,*values):
            key=(operation,*(v.as_pointer() if hasattr(v,'as_pointer') else ('n',float(v)) for v in values))
            if key in self.cache: return self.cache[key]
            node=self.tree.nodes.new('ShaderNodeMath'); node.operation=operation;node.use_clamp=False
            node.hide=True;node.label=operation.title();self.index+=1
            node.location=(-950+(self.index%16)*135, -500-(self.index//16)*55)
            for socket,value in zip(node.inputs,values):
                if hasattr(value,'as_pointer'): self.tree.links.new(value,socket)
                else: socket.default_value=float(value)
            result=node.outputs[0]; self.cache[key]=result; return result

        def add(self,a,b):
            if isinstance(a,(int,float)) and a==0:return b
            if isinstance(b,(int,float)) and b==0:return a
            return self.op('ADD',a,b)

        def sub(self,a,b):
            if isinstance(b,(int,float)) and b==0:return a
            return self.op('SUBTRACT',a,b)

        def mul(self,a,b):
            if isinstance(a,(int,float)):
                if a==0:return 0.
                if a==1:return b
            if isinstance(b,(int,float)):
                if b==0:return 0.
                if b==1:return a
            return self.op('MULTIPLY',a,b)

        def div(self,a,b):return self.op('DIVIDE',a,b)

        def power(self,a,b):return self.op('POWER',a,b)

        def max(self,a,b):return self.op('MAXIMUM',a,b)

        def min(self,a,b):return self.op('MINIMUM',a,b)

        def exp(self,a):return self.op('EXPONENT',a)

        def sum(self,*values):
            if not values:return 0.
            a=values[0]
            for b in values[1:]:a=self.add(a,b)
            return a

        def lt(self,a,b):return self.op('LESS_THAN',a,b)

        def vector(self,values):
            key=('VECTOR',*(v.as_pointer() if hasattr(v,'as_pointer') else ('n',float(v)) for v in values))
            if key in self.cache:return self.cache[key]
            n=self.tree.nodes.new('ShaderNodeCombineXYZ');n.hide=True
            for s,v in zip(n.inputs,values):self.tree.links.new(v,s) if hasattr(v,'as_pointer') else setattr(s,'default_value',v)
            self.cache[key]=n.outputs[0];return n.outputs[0]

        def split(self,vector):
            key=('SPLIT',vector.as_pointer())
            if key in self.cache:return self.cache[key]
            n=self.tree.nodes.new('ShaderNodeSeparateXYZ');n.hide=True;self.tree.links.new(vector,n.inputs[0])
            self.cache[key]=tuple(n.outputs);return tuple(n.outputs)


    class Graph(_Graph):
        size=cm.SIZE
        planes=1

        def op(self,operation,*values):
            # Constant-only nodes interrupt compositor pixel compile units. Fold
            # their socket-precision arithmetic before constructing the address
            # kernel, so large local indices remain shader registers.
            if operation in {'ADD','SUBTRACT','MULTIPLY','DIVIDE'} and all(isinstance(v,(int,float)) for v in values):
                f32 = _application.component('graph.optimize').f32
                a,b=map(f32,values)
                value={'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,
                       'DIVIDE':lambda:a/b if b else 0.}[operation]()
                return f32(value)
            return super().op(operation,*values)

        def encode(self,x):
            t=self.div(x,cm.KAPPA)
            v=self.op('LOGARITHM',self.add(1,t),np.e)
            return self.div(v,cm.LOG_RANGE)

        def cell(self,x):
            # Input is repaired before addressing. One exponential recovers both
            # endpoints; bound fraction rounding so interpolation never extrapolates.
            bounded=self.min(cm.MAXIMUM,self.max(cm.MINIMUM,x))
            cells=self.size-1
            step=cm.LOG_RANGE/cells
            position=self.mul(cells,self.encode(bounded))
            i=self.min(cells-1,self.max(0,self.op('FLOOR',position)))
            e=self.exp(self.mul(i,step))
            lo=self.mul(cm.KAPPA,self.sub(e,1))
            hi=self.mul(cm.KAPPA,self.sub(self.mul(e,math.exp(step)),1))
            fraction=self.min(1,self.max(0,self.div(self.sub(x,lo),self.sub(hi,lo))))
            return i,self.max(fraction,self.sub(1,self.lt(x,cm.MAXIMUM)))

        def uv(self,index):
            row=self.op('FLOOR',self.div(index,cm.ATLAS_WIDTH))
            col=self.sub(index,self.mul(cm.ATLAS_WIDTH,row))
            return self.vector((self.div(self.add(col,.5),cm.ATLAS_WIDTH),
                                self.div(self.add(row,.5),cm.atlas_height(self.size,self.planes)),1))

        def sample(self,index):
            return self.sample_uv(self.uv(index))

        def sample_uv(self,uv):
            node=self.tree.nodes.new('ShaderNodeTexImage');node.interpolation='Closest';node.extension='EXTEND'
            self.tree.links.new(uv,node.inputs['Vector']);value=node.outputs['Color']
            node['_pg_cube_image']=True
            node.hide=True;node.label='LUT tetrahedral vertex'
            return self.split(value)

        def coordinates(self,rgb,edges=False):
            cells=[self.cell(x) for x in self.split(rgb)]
            low=[i for i,f in cells];f=[f for i,f in cells];x,y,z=f
            ge=lambda a,b:self.sub(1,self.lt(a,b))
            high_axis=(self.mul(ge(x,y),ge(x,z)),self.mul(self.lt(x,y),ge(y,z)),self.mul(self.lt(x,z),self.lt(y,z)))
            low_axis=(self.mul(self.lt(x,y),self.lt(x,z)),self.mul(ge(x,y),self.lt(y,z)),self.mul(ge(x,z),ge(y,z)))
            stride=(self.size**2,self.size,1);diagonal=sum(stride)
            base=self.sum(*(self.mul(i,k) for i,k in zip(low,stride)))
            first=self.sum(*(self.mul(i,k) for i,k in zip(high_axis,stride)))
            last=self.sum(*(self.mul(i,k) for i,k in zip(low_axis,stride)))
            maximum=self.max(x,self.max(y,z));minimum=self.min(x,self.min(y,z))
            middle=self.min(self.max(x,y),self.max(self.min(x,y),z))
            indices=(base,self.add(base,first),self.sub(self.add(base,diagonal),last),self.add(base,diagonal))
            if edges:
                # Plane 0 stores base RGB. Planes 1..3 store forward R/G/B
                # differences at each lattice point. Select the ordered edges
                # without subtracting half-float neighboring vertices at lookup.
                high_plane=self.sum(*(self.mul(k,a) for k,a in zip((1,2,3),high_axis)))
                low_plane=self.sum(*(self.mul(k,a) for k,a in zip((1,2,3),low_axis)))
                middle_plane=self.sub(self.sub(6,high_plane),low_plane)
                count=self.size**3
                indices=(base,self.add(base,self.mul(count,high_plane)),
                         self.add(indices[1],self.mul(count,middle_plane)),
                         self.add(indices[2],self.mul(count,low_plane)))
            return indices,(maximum,middle,minimum)

        def lookup(self,rgb):
            indices,fractions=self.coordinates(rgb)
            samples=[self.sample(i) for i in indices]
            return self.blend(samples,fractions)

        def blend(self,vertices,fractions):
            result=[]
            for c in range(3):
                value=vertices[0][c]
                for a,b,f in zip(vertices,vertices[1:],fractions):
                    value=self.op('MULTIPLY_ADD',f,self.sub(b[c],a[c]),value)
                result.append(value)
            return self.vector(result)


    def compositor_helper(kind,size=cm.SIZE,tree_type='CompositorNodeTree'):
        """Keep address arithmetic/linear blending inside float32 shader units.

        Auto still uses half-float *images*. Only UVs, physical fractions and RGBE
        base/edge coefficients cross boundaries. Small edge slopes are encoded
        independently; decoded HDR coefficients stay inside float32 shader units.
        """
        tag=f'{SCHEMA}:{kind}:{size}'
        for tree in bpy.data.node_groups:
            if tree.bl_idname==tree_type and tree.get('_pg_cube_op')==tag:return tree
        private_name = _application.component('graph.presentation').private_name
        tree=bpy.data.node_groups.new(private_name('Perceptual Grade · Cube '+kind),tree_type)
        if kind=='Address':
            inputs=[('RGB','NodeSocketVector')]
            outputs=[*(('UV'+str(i),'NodeSocketVector') for i in range(4)),('Fractions','NodeSocketVector')]
        else:
            inputs=[*((f'RGB{i}','NodeSocketVector') for i in range(4)),
                    *((f'E{i}','NodeSocketFloat') for i in range(4)),('Fractions','NodeSocketVector')]
            outputs=[('RGB','NodeSocketVector')]
        for direction,items in (('INPUT',inputs),('OUTPUT',outputs)):
            for name,typ in items:tree.interface.new_socket(name=name,in_out=direction,socket_type=typ)
        source=tree.nodes.new('NodeGroupInput');target=tree.nodes.new('NodeGroupOutput');target.is_active_output=True
        g=Graph(tree,source);g.size=size
        if kind=='Address':
            packed=tree_type=='CompositorNodeTree'
            g.planes=4 if packed else 1
            indices,fractions=g.coordinates(source.outputs['RGB'],edges=packed)
            values=[*(g.uv(index) for index in indices),g.vector(fractions)]
        else:
            fractions=g.split(source.outputs['Fractions'])
            rgb=[g.split(source.outputs[f'RGB{i}']) for i in range(4)]
            factors=[g.power(2,source.outputs[f'E{i}']) for i in range(4)]
            decoded=[[g.mul(f,r[c]) for c in range(3)] for f,r in zip(factors,rgb)]
            result=[]
            for c in range(3):
                value=decoded[0][c]
                for edge,f in zip(decoded[1:],fractions):
                    value=g.op('MULTIPLY_ADD',f,edge[c],value)
                result.append(value)
            values=[g.vector(result)]
        for value,socket in zip(values,target.inputs):tree.links.new(value,socket)
        remove_dead_nodes = _application.component('graph.optimize').remove_dead_nodes; fold_multiply_adds = _application.component('graph.optimize').fold_multiply_adds
        remove_dead_nodes(tree);fold_multiply_adds(tree);tree['_pg_cube_op']=tag
        return tree


    def compositor_lookup(tree,rgb,size=cm.SIZE,slot=0,image=None):
        address=tree.nodes.new('CompositorNodeGroup');address.node_tree=compositor_helper('Address',size);address.hide=True
        address['_pg_cube_address']=True
        address[resources.SLOT]=slot
        tree.links.new(rgb,address.inputs['RGB'])
        blend=tree.nodes.new('CompositorNodeGroup');blend.node_tree=compositor_helper('Blend');blend.hide=True
        tree.links.new(address.outputs['Fractions'],blend.inputs['Fractions'])
        texture=tree.nodes.new('CompositorNodeImage');texture['_pg_cube_image']=True;texture.hide=True
        texture[resources.SLOT]=slot;texture.image=image
        for i in range(4):
            sample=tree.nodes.new('CompositorNodeMapUV');sample.hide=True;sample.label='LUT tetrahedral vertex'
            sample.inputs['Interpolation'].default_value='Nearest'
            sample.inputs['Extension X'].default_value='Extend';sample.inputs['Extension Y'].default_value='Extend'
            tree.links.new(texture.outputs['Image'],sample.inputs['Image']);tree.links.new(address.outputs[f'UV{i}'],sample.inputs['UV'])
            split=tree.nodes.new('CompositorNodeSeparateColor');split.mode='RGB';split.hide=True
            tree.links.new(sample.outputs['Image'],split.inputs['Image'])
            tree.links.new(sample.outputs['Image'],blend.inputs[f'RGB{i}']);tree.links.new(split.outputs['Alpha'],blend.inputs[f'E{i}'])
        return blend.outputs['RGB']


    def prepare_helpers(tree_type):
        """Build shared address groups during setup, before a preview is ready."""
        for size in (cm.SIZE,cm.PREVIEW_SIZE):
            compositor_helper('Address',size,tree_type)


    def shader_lookup(tree,source,rgb,size=cm.SIZE,slot=0,image=None):
        """Shared address kernel for native Shader LUT sampling."""
        address=tree.nodes.new('ShaderNodeGroup');address.hide=True
        address.node_tree=compositor_helper('Address',size,tree_type='ShaderNodeTree')
        address['_pg_cube_address']=True
        address[resources.SLOT]=slot
        tree.links.new(rgb,address.inputs['RGB'])
        g=Graph(tree,source);g.size=size
        fractions=g.split(address.outputs['Fractions'])
        before={n.as_pointer() for n in tree.nodes}
        samples=[g.sample_uv(address.outputs['UV'+str(i)]) for i in range(4)]
        for texture in tree.nodes:
            if texture.as_pointer() not in before and texture.get('_pg_cube_image'):
                texture[resources.SLOT]=slot;texture.image=image
        return g.blend(samples,fractions)


    def repair_input(tree,source):
        """Native Stage 1, then AP1 repair, without overflowing signed RGB.

        Carry the raw amplitude separately through the matrix. Normalization is
        algebraic evaluation of the same linear map, before its input projection.
        """
        spaces = _application.component('graph.spaces')
        g=_Graph(tree,None)
        r=g.split(source.outputs['Color'])
        magnitude=g.max(g.op('ABSOLUTE',r[0]),g.max(g.op('ABSOLUTE',r[1]),g.op('ABSOLUTE',r[2])))
        unit_rgb=g.vector([g.div(x,magnitude) for x in r])
        unit=g.split(spaces.multiply(tree,unit_rgb,spaces.LUT_INPUT_ROWS))
        y=g.max(0,g.sum(*(g.mul(float(w),x) for w,x in zip(cm.AP1_Y,unit))))
        k=g.max(0,g.mul(-1,g.min(unit[0],g.min(unit[1],unit[2]))))
        scale=g.div(y,g.add(y,k))
        lifted=[g.mul(scale,g.add(x,k)) for x in unit]
        maximum=g.max(lifted[0],g.max(lifted[1],lifted[2]))
        factor=g.min(magnitude,g.div(cm.MAXIMUM,maximum))
        mapped=[g.mul(x,factor) for x in lifted]
        return g.vector(mapped)


    def _make_bank(size,encoded,index,pixels=None):
        """Structural allocation only; called before its owner is published."""
        image=bpy.data.images.new(tr('datablock.color_lut')+f' {size} {index%2}',
            width=cm.ATLAS_WIDTH,height=cm.atlas_height(size,4 if encoded else 1),
            float_buffer=True,alpha=True)
        try:
            image.colorspace_settings.name='Non-Color';image.alpha_mode='CHANNEL_PACKED'
            image.use_half_precision=False;image.file_format='OPEN_EXR'
            image['_pg_cube_schema']=SCHEMA;image['_pg_cube_size']=size
            image['_pg_cube_encoding']='RGBE_DIFFERENCES' if encoded else 'RGB'
            image[resources.SLOT]=index
            # Zero is a readable neutral pending payload for both encodings. Completion
            # authority is deliberately absent until an admitted numerical result lands.
            values=(np.zeros(len(image.pixels),np.float32) if pixels is None
                    else resources.validate_pixels(image,pixels))
            image.pixels.foreach_set(values);image.update()
            resources.pack_image(image)
            return image
        except Exception:
            if image.users==0:remove_image(image)
            raise


    def clone_kernel(tree):
        """Private writable storage, exclusively inside a structural operation."""
        if tree.get('_pg_cube_schema')!=SCHEMA or tree.get(resources.LAYOUT_KEY)!=resources.LAYOUT:
            raise ValueError('Cannot copy an unsupported output kernel')
        copied=tree.copy();created=[]
        try:
            images={}
            for texture in copied.nodes:
                if not texture.get('_pg_cube_image'):continue
                index=int(texture[resources.SLOT])
                if index not in images:
                    source=texture.image
                    pixels=np.empty(len(source.pixels),np.float32);source.pixels.foreach_get(pixels)
                    image=_make_bank(int(source['_pg_cube_size']),
                        source['_pg_cube_encoding']=='RGBE_DIFFERENCES',index,pixels)
                    created.append(image)
                    for key in source.keys():
                        value=source[key]
                        if type(value) in (str,bool,int,float):image[key]=value
                    images[index]=image
                texture.image=images[index]
            # Baked raster Images are intentionally shared; only numerical
            # payload banks above are private to the copied Grade/Stack.
            return copied
        except Exception:
            remove_node_group = _application.app.blender.native.remove_node_group
            remove_node_group(copied)
            for image in created:
                if image.users==0:remove_image(image)
            raise


    def make_kernel(tree_type,kind='STACK'):
        """Construct a complete unpublished owner/target unit, or remove it all."""
        created=[]
        try:
            return _build_kernel(tree_type,kind,created)
        except Exception:
            remove_node_group = _application.app.blender.native.remove_node_group
            trees=[value for kind_,value in created if kind_=='tree']
            images=[value for kind_,value in created if kind_=='image']
            for abandoned_tree in trees:
                if abandoned_tree.users==0:remove_node_group(abandoned_tree)
            for abandoned_image in images:
                if abandoned_image.users==0:remove_image(abandoned_image)
            raise


    def _build_kernel(tree_type,kind,created):
        """Nonlinear Grade LUT or complete Stack LUT, with private native rows."""
        nodes = _application.component('graph.nodes')
        spaces = _application.component('graph.spaces')
        private_name = _application.component('graph.presentation').private_name
        prepare_helpers(tree_type)
        tree=bpy.data.node_groups.new(private_name('Perceptual Grade · '+('Grade' if kind=='FORWARD' else 'Stack')),tree_type)
        created.append(('tree',tree))
        tree[nodes.KERNEL]=nodes.VERSION;tree['_pg_cube_schema']=SCHEMA
        tree[resources.LAYOUT_KEY]=resources.LAYOUT
        tree['_pg_transform_kind']=kind
        shader=tree_type=='ShaderNodeTree'
        for direction in ('INPUT','OUTPUT'):
            tree.interface.new_socket(name='Color',in_out=direction,socket_type='NodeSocketColor')
            if shader:
                socket=tree.interface.new_socket(name='Alpha',in_out=direction,socket_type='NodeSocketFloat')
                socket.default_value=1.
        if kind=='FORWARD':
            socket=tree.interface.new_socket(name='Color Reference',in_out='INPUT',socket_type='NodeSocketColor')
            socket.hide_value=True
        spaces.constants(tree,spaces.LUT_INPUT_ROWS)
        spaces.constants(tree,spaces.LUT_OUTPUT_ROWS)
        selected=resources.create_selector(tree)
        source=tree.nodes.new('NodeGroupInput');output=tree.nodes.new('NodeGroupOutput')
        output.is_active_output=True
        rgb=repair_input(tree,source)
        g=_Graph(tree,source)
        components=[]
        for index,size in enumerate((cm.PREVIEW_SIZE,cm.PREVIEW_SIZE,cm.SIZE,cm.SIZE)):
            image=_make_bank(size,not shader,index)
            created.append(('image',image))
            value=(shader_lookup(tree,source,rgb,size,index,image) if shader
                   else compositor_lookup(tree,rgb,size,index,image))
            weight=g.op('COMPARE',selected,index,0.1)
            components.append([g.mul(component,weight) for component in g.split(value)])
        result=g.vector([g.sum(*(value[c] for value in components)) for c in range(3)])
        result=spaces.multiply(tree,result,spaces.LUT_OUTPUT_ROWS)
        pending=spaces.multiply(tree,rgb,spaces.LUT_OUTPUT_ROWS)
        route=tree.nodes.new('NodeReroute');route.name=ROUTE
        bypass=g.lt(selected,0.)
        combined=g.vector([g.add(a,g.mul(b,bypass)) for a,b in zip(g.split(result),g.split(pending))])
        tree.links.new(combined,route.inputs[0])
        if shader:
            tree.links.new(route.outputs[0],output.inputs['Color'])
            tree.links.new(source.outputs['Alpha'],output.inputs['Alpha'])
        else:
            separate=tree.nodes.new('CompositorNodeSeparateColor');separate.mode='RGB'
            tree.links.new(source.outputs['Color'],separate.inputs['Image'])
            components=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(route.outputs[0],components.inputs[0])
            combine=tree.nodes.new('CompositorNodeCombineColor');combine.mode='RGB'
            for value,name in zip(components.outputs,('Red','Green','Blue')):
                tree.links.new(value,combine.inputs[name])
            tree.links.new(separate.outputs['Alpha'],combine.inputs['Alpha'])
            tree.links.new(combine.outputs['Image'],output.inputs['Color'])
        _application.component('graph.bake').create_output(tree)
        return tree


    def select(node,use_cube):
        if is_adopted(node) or not resources.is_current(node):return
        if not resources.unique_owner(node):
            raise RuntimeError('Native output constants require private writable resources')
        state=resources.record(node.node_tree,node)['state']
        index=state.selected if use_cube else -1
        resources.select_payload(node,index)


    def image_for(node):
        # Retain every wrapper until this lookup ends. RNA's values() constructs
        # a list in native code; list(collection) would use its Python iterator.
        return resources.current_image(node)


    def adopt_native(node, intent):
        stop(_states.pop(node.as_pointer(), None))
        resources.adopt_native(node, intent)


    def is_adopted(node):
        return resources.is_adopted(node)


    def has_runtime_intent(node):
        return resources.has_runtime_intent(node)


    def request_intent(node, intent, explicit=False):
        return resources.request_intent(node, intent, explicit)


    def read_cube(image, verified=None):
        """Return only this runtime's produced Cube; restored pixels are external."""
        if image is None:return None
        cached=_cubes.get(image.as_pointer())
        if (cached is not None and resolve(cached[0])==image
                and cached[1].signature==image.get('_pg_cube_signature')
                and cached[2]==image.get(resources.REVISION)):
            return cached[1]
        return None


    def installed_cube(node):
        """Only the LUT connected to the native output is a scope source."""
        image=image_for(node)
        cube=read_cube(image) if image is not None else None
        return cube


    def installed_signal(node):
        """Snapshot the installed cube and the native float32 socket matrices."""
        spaces = _application.component('graph.spaces')
        cube=installed_cube(node)
        before=np.array(spaces.read_rows(node,spaces.LUT_INPUT_ROWS))
        after=np.array(spaces.read_rows(node,spaces.LUT_OUTPUT_ROWS))
        return cube,before,after


    def signal_signature(node):
        """Installed image and committed rows, without decoding image pixels."""
        spaces = _application.component('graph.spaces')
        image=image_for(node)
        token=None if image is None else (image.as_pointer(),image.get('_pg_cube_signature'),image.get('_pg_cube_schema'),image.get('_pg_cube_size'))
        rows=spaces.read_rows(node)
        return token,rows


    def scope_rows(node,model=None):
        """Freeze the exact float32 endpoints independently of preview installation."""
        nodes = _application.component('graph.nodes'); spaces = _application.component('graph.spaces')
        model=nodes.bake_model(node) if model is None else model
        before,after=spaces.adapters(nodes.working_space())
        if nodes.is_grade(node):
            first=cm.TO_LUT@nodes.linear_matrix(node)@before;last=after@cm.FROM_LUT
        else:
            first=model.input_matrix@cm.TO_LUT@before;last=after@cm.FROM_LUT@model.output_matrix
        result=tuple(np.array(value,dtype='f4') for value in (first,last))
        for value in result:value.flags.writeable=False
        return result


    def scope_signature(node,model=None,rows=None):
        nodes = _application.component('graph.nodes')
        model=nodes.bake_model(node) if model is None else model
        rows=scope_rows(node,model) if rows is None else rows
        return cm.signature(model),tuple(tuple(float(v) for v in row) for matrix in rows for row in matrix)


    def _remember_final(cube):
        if cube is None or cube.size!=cm.SIZE:return
        cube.values.flags.writeable=False
        _ready[cube.signature]=cube;_ready.move_to_end(cube.signature)
        # Active demand owners hold direct references once captured. Pending demand
        # signatures remain pinned until their complete dependency snapshot is read.
        for key in tuple(_ready):
            if len(_ready)<=8:break
            if key not in _scope_demands:_ready.pop(key,None)


    def _scope_owner(node,owner):
        return ('node',node.as_pointer()) if owner is None else ('snapshot',owner)


    def _scope_drop_owner(key,token):
        demand=_scope_demands.get(key)
        if demand is None:return
        demand['owners'].pop(token,None)
        if not demand['owners']:_scope_demands.pop(key,None)


    def scope_signal(node,model=None,rows=None,owner=None):
        """Read only a 65³ signal; demand a frozen model without installing it."""
        nodes = _application.component('graph.nodes')
        model=nodes.bake_model(node) if model is None else model
        key=cm.signature(model);rows=scope_rows(node,model) if rows is None else rows
        token=_scope_owner(node,owner)
        cube=_ready.get(key)
        if cube is None and not node.get(INCOMPLETE):
            try:
                manifest=_manifest(node)
                if manifest['size']==cm.SIZE and manifest['signature']==key:
                    installed=installed_cube(node)
                    if installed is not None and installed.size==cm.SIZE and installed.signature==key:
                        cube=installed;_remember_final(cube)
            except (ValueError,TypeError,KeyError,ReferenceError,RuntimeError,StopIteration):pass
        if cube is not None:
            _scope_drop_owner(key,token)
            return cube,*rows
        if key in _scope_failures:raise ScopeFailure(_scope_failures[key])
        if key not in _scope_demands:
            _scope_demands[key]=dict(owners=OrderedDict())
        _scope_demands[key]['owners'][token]=dict(node=retain(node,'lut.scope_demand'),model=model,context=context(node))
        scheduler.wake(scope_tick,.001)
        scheduler.wake(tick,.001)
        raise ScopePreparing('Preparing 65³ scopes')


    def scope_release(node,model=None,owner=None):
        nodes = _application.component('graph.nodes')
        key=cm.signature(nodes.bake_model(node) if model is None else model)
        _scope_drop_owner(key,_scope_owner(node,owner))


    def scope_retry():
        """Explicit Retry, never a poll-driven retry of unchanged failed inputs."""
        _scope_failures.clear()
        _failures.clear()


    def _scope_protected(state):
        return state['baker'].signature in _scope_demands


    def scope_tick(start=True):
        """Finish frozen final demands using the shared cache and physical CPU lane.

        A matching native baker is reused, including its GPU path. A detached
        demand only runs when no native CPU job occupies the lane; subsequent
        native jobs wait for it. No duplicate bake is queued for a running key.
        """
        nonlocal _scope_running
        if not scheduler.lifecycle_admits('start'):return .016
        get_runtime = _application.component('services').get_runtime
        get_runtime().poll()
        collect_retired()
        if _scope_running is not None:
            key,baker,future,ticket=_scope_running
            if not future.done():return .016
            _scope_running=None
            try:
                scheduler.effect(ticket,'worker',future.result)
                if baker.cube is not None and baker.cube.size==cm.SIZE and scheduler.publication(ticket,commit=False):
                    _remember_final(baker.cube)
                    scheduler.publication(ticket)
                else:
                    _scope_failures[key]='The final scope bake was superseded before completion'
            except CancelledError:pass
            except Exception as exc:
                _scope_failures[key]=diagnostic_text(exc);_scope_demands.pop(key,None)
                log_exception(LOG,'Perceptual Grade final scope bake failed')
            finally:
                scheduler.scope_complete(ticket)
                scheduler.scope_bake_clear(key)
        if not start:return None
        for key,demand in tuple(_scope_demands.items()):
            if key in _ready:continue
            live=None
            for token,request in tuple(demand['owners'].items()):
                node=resolve(request['node'])
                try:
                    sampling = _application.component('scopes.sampling')
                    if sampling.live(node) and context(node)==request['context']:
                        live=request;break
                except (ReferenceError,RuntimeError):pass
                _scope_drop_owner(key,token)
            if live is None:continue
            node=resolve(live['node'])
            matches=[state for state in _states.values()
                     if state['baker'].signature==key and not state['superseded']]
            if matches:
                for state in matches:_remember_final(state['baker'].cube)
                continue
            if _retired or any(state.get('future') is not None for state in _states.values()):return .016
            ticket=None
            try:
                nodes = _application.component('graph.nodes')
                ticket=scheduler.scope_bake_request(key)
                if not scheduler.scope_start(ticket):continue
                baker=cpu_baker(live['model'],False,node.node_tree.bl_idname=='CompositorNodeTree',not nodes.is_grade(node))
                future=scheduler.effect(ticket,'submit',baker.start)
                _scope_running=(key,baker,future,ticket)
            except Exception as exc:
                if ticket is not None:scheduler.scope_failed(ticket,exc)
                _scope_failures[key]=diagnostic_text(exc);_scope_demands.pop(key,None)
                log_exception(LOG,'Perceptual Grade final scope submission failed')
            return .016
        return .016 if any(key not in _ready for key in _scope_demands) else None


    def evaluate_signal(rgb,signal):
        cube,before,after=signal
        if cube is None:raise ValueError(tr('scope.stack.waiting'))
        return cm.tetrahedral(cube.values,np.asarray(rgb)@before.T)@after.T


    def uses_lut(node):
        nodes = _application.component('graph.nodes'); stacks = _application.component('graph.stacks')
        return (nodes.is_grade(node) or stacks.is_stack(node)) and resources.is_current(node)


    def baked_only(node):
        return uses_lut(node)


    def stop(state):
        if state is None:return
        state['baker'].cancel()
        futures=([state['future']] if state.get('future') is not None else [])
        if getattr(state['baker'],'handle',None) is not None and state['baker'] not in futures:
            futures.append(state['baker'])
        futures.extend(future for _,future in getattr(state['baker'],'packing',()))
        for future in futures:
            future.cancel()
            if not _physically_done(future) and future not in _retired:
                _retired.add(future)
        ticket=state.get('schedule')
        if ticket is not None:
            if any(not _physically_done(future) for future in futures):
                _retirements[ticket]=tuple(futures)
            else:
                scheduler.scope_complete(ticket)


    def _physically_done(future):
        retired=getattr(future,'retired',None)
        return bool(retired() if callable(retired) else future.done())


    def collect_retired():
        """Complete policy slots on the main thread after physical work ends."""
        for ticket,futures in tuple(_retirements.items()):
            if all(_physically_done(future) for future in futures):
                scheduler.scope_complete(ticket)
                _retirements.pop(ticket,None)
        _retired.difference_update(future for future in tuple(_retired) if _physically_done(future))
        return not (_retirements or _retired)


    def _finished(state):
        """Whether the adapter can safely advance an obsolete bake snapshot.

        A canceled consumer can finish before its helper acknowledges retirement.
        Keep the coordinator lane occupied until physical execution has ended.
        """
        baker=state['baker']
        if getattr(baker,'handle',None) is not None and not baker.done():return False
        if any(not future.done() for _,future in getattr(baker,'packing',())):
            return False
        if state['superseded']:
            return state.get('future') is None
        if state.get('future') is not None:
            return False
        if baker.requires_main_thread:
            return bool(baker.calculated and not baker.packing and baker.cube is not None)
        return baker.cube is not None


    def context(node):
        """Changes of owner, scene space or Stack topology invalidate old previews."""
        nodes = _application.component('graph.nodes')
        topology=() if nodes.is_grade(node) else tuple((item.key,item.enabled,item.direction,
            item.grade.as_pointer() if item.grade else 0) for item in node.pg_stack_entries)
        return node.node_tree.as_pointer(),nodes.working_space(),topology


    def commit_rows(node,model):
        """Commit independent Grade Stage 1, or a coherent Stack snapshot's endpoints."""
        nodes = _application.component('graph.nodes'); spaces = _application.component('graph.spaces')
        before,after=spaces.adapters(nodes.working_space())
        if nodes.is_grade(node):
            first=cm.TO_LUT@nodes.linear_matrix(node)@before;last=after@cm.FROM_LUT
        else:
            first=model.input_matrix@cm.TO_LUT@before;last=after@cm.FROM_LUT@model.output_matrix
        return commit_endpoint_rows(node,first,last)


    def _checked_rows(before,after):
        values=np.asarray((before,after),dtype=np.float32)
        if values.shape!=(2,3,3) or not np.isfinite(values).all() or np.any(np.abs(values)>2.**120):
            raise ValueError('Invalid LUT endpoint matrices')
        return tuple(tuple(float(v) for v in row) for matrix in values for row in matrix)


    def _write_rows(node,rows):
        spaces = _application.component('graph.spaces'); nodes = _application.component('graph.nodes')
        names=(*spaces.LUT_INPUT_ROWS,*spaces.LUT_OUTPUT_ROWS)
        if (len(rows)!=len(names) or not np.isfinite(rows).all()
                or np.any(np.abs(rows)>2.**120)):
            raise ValueError('Invalid LUT endpoint rows')
        spaces.write_rows(node,rows)
        if spaces.read_rows(node)!=tuple(rows):
            raise ValueError('LUT endpoint readback differs')


    def _output_key(manifest):
        return json.dumps(manifest,sort_keys=True,separators=(',',':'))


    def _fallback(node):
        resources.select_payload(node,-1)
        if resources.selected_index(node)!=-1:raise RuntimeError('LUT fallback could not be verified')


    def _install_graph(node,index,rows,*,checkpoints=False,defer=False):
        """Commit a complete output; all graph mutations preserve ID relationships."""
        image=resources.bank_images(node)[index] if index>=0 else None
        source=image.get('_pg_cube_signature') if image is not None else 0
        # The intent includes the exact float32 endpoint rows, not just a bake key.
        expected=(validate_output(node,_candidate={'index':index,'rows':rows}) if index>=0 else None)
        key=_output_key(expected) if expected is not None else 0
        token=resources.graph_begin(node,index,source,key)
        current=resources.record(node.node_tree,node)
        current['graph_token']=token
        node[INCOMPLETE]=True
        _fallback(node)
        resources.graph_receipt(node,'fallback',token,validated=True)
        _write_rows(node,rows)
        if checkpoints:_publication_checkpoint('rows')
        manifest=validate_output(node,_candidate={'index':index}) if index>=0 else None
        if manifest!=expected:raise ValueError('LUT graph differs from the reserved output')
        if checkpoints:_publication_checkpoint('validated')
        if manifest is not None:node[MANIFEST]=json.dumps(manifest,sort_keys=True,separators=(',',':'))
        elif MANIFEST in node:del node[MANIFEST]
        if checkpoints:_publication_checkpoint('manifest')
        resources.graph_receipt(node,'rows',token,validated=True)
        if index>=0:
            resources.select_payload(node,index)
            if resources.selected_index(node)!=index:raise RuntimeError('LUT selector readback differs')
            if checkpoints:_publication_checkpoint('selector')
            if INCOMPLETE in node:del node[INCOMPLETE]
            if defer:return manifest
            resources.graph_receipt(node,'select',token,validated=True)
        else:
            resources.graph_receipt(node,'fallback_done',token,validated=True)
        current.pop('graph_token',None)
        return manifest


    def commit_endpoint_rows(node,before,after):
        """Matrix-only edits share publication fencing without recomputing pixels."""
        if is_adopted(node):return True
        output_guard = _application.component('graph.output_guard'); spaces = _application.component('graph.spaces')
        rows=_checked_rows(before,after)
        names=(*spaces.LUT_INPUT_ROWS,*spaces.LUT_OUTPUT_ROWS)
        if spaces.read_rows(node)==rows:return True
        if not resources.unique_owner(node):
            raise RuntimeError('Native output constants require private writable resources')
        if not output_guard.publishing():
            if not output_guard.available():return False
            with output_guard.publication():return commit_endpoint_rows(node,before,after)
        _repair_graph(node)
        current=resources.record(node.node_tree,node)
        state=current['state'];index=state.selected
        if index>=0 and not resources.policy.complete(state,index,state.slots[index].source):index=-1
        snapshot=_graph_snapshot(node)
        try:
            _install_graph(node,index,rows)
        except Exception:
            _rollback_graph(node,snapshot)
            raise
        return True


    MANIFEST='_pg_output_manifest'
    INCOMPLETE='_pg_output_pending'


    def refresh_manifest(node):
        """Keep saved output metadata aligned with independent endpoint edits."""
        if MANIFEST not in node or node.get(INCOMPLETE):return
        try:
            recorded=json.dumps(_manifest(node),sort_keys=True,separators=(',',':'))
            if node[MANIFEST]!=recorded:node[MANIFEST]=recorded
        except (ValueError,TypeError,KeyError,ReferenceError,RuntimeError,StopIteration):
            # Save/load validation will reject and rebuild incomplete graphs.
            pass


    def _output_snapshot(node):
        """A bounded owner for RNA wrappers, used only during one synchronous read."""
        tree=node.node_tree
        kernel_nodes=tree.nodes.values()
        images=[]
        for item in kernel_nodes:
            if item.get('_pg_cube_image'):
                images.append((item,item.image))
        return tree,kernel_nodes,images


    def _manifest(node,_snapshot=None,_candidate=None):
        """Portable bank description, backed by this incarnation's payload receipt."""
        spaces = _application.component('graph.spaces')
        tree,kernel_nodes,images=_output_snapshot(node) if _snapshot is None else _snapshot
        if not resources.is_current(node):raise ValueError('Invalid output resource layout')
        index=resources.selected_index(node) if _candidate is None else _candidate['index']
        if index<0:raise ValueError('No completed LUT output is selected')
        banks=resources.bank_images(node,(tree,kernel_nodes,images))
        image=banks[index];state=resources.record(tree,node)['state']
        if _candidate is None and (state.selected!=index or not state.graph_valid):
            raise ValueError('LUT output has no current completion receipt')
        signature=image.get('_pg_cube_signature')
        if not resources.policy.complete(state,index,signature):
            raise ValueError('LUT payload generation requires reconstruction')
        encoding='RGBE_DIFFERENCES' if tree.bl_idname=='CompositorNodeTree' else 'RGB'
        for bank,storage in banks.items():
            size=cm.PREVIEW_SIZE if bank<2 else cm.SIZE
            if (storage.get('_pg_cube_schema')!=SCHEMA or storage.get('_pg_cube_size')!=size
                    or storage.get('_pg_cube_encoding')!=encoding
                    or tuple(storage.size)!=(cm.ATLAS_WIDTH,cm.atlas_height(size,4 if encoding=='RGBE_DIFFERENCES' else 1))):
                raise ValueError('Invalid permanent LUT bank layout')
        address_slots=set()
        for item in kernel_nodes:
            if item.get('_pg_cube_address'):
                bank=item.get(resources.SLOT);size=cm.PREVIEW_SIZE if bank<2 else cm.SIZE
                address=item.node_tree
                if address is None or address.get('_pg_cube_op')!=f'{SCHEMA}:Address:{size}':
                    raise ValueError('Incompatible permanent LUT address helper')
                address_slots.add(bank)
        if address_slots!={0,1,2,3}:raise ValueError('Missing permanent LUT address helper')
        planned=_candidate.get('rows') if _candidate is not None else None
        rows=([list(row) for row in planned] if planned is not None else
              [list(row) for row in spaces.read_rows(node)])
        if not np.isfinite(rows).all() or np.any(np.abs(rows)>2.**120):raise ValueError('Invalid LUT endpoints')
        manifest=dict(version=2,schema=SCHEMA,layout=resources.LAYOUT,slot=index,
            signature=signature,size=image['_pg_cube_size'],encoding=encoding,
            revision=image.get(resources.REVISION),rows=rows,route=ROUTE)
        if _candidate is None and not resources.policy.output_complete(state,_output_key(manifest)):
            raise ValueError('LUT graph no longer matches its completion receipt')
        return manifest


    def validate_output(node,model=None,final=False,*,_snapshot=None,_candidate=None):
        manifest=_manifest(node,_snapshot,_candidate)
        if final and manifest['size']!=cm.SIZE:raise ValueError('Final LUT is not 65³')
        if model is not None:
            if manifest['signature']!=cm.signature(model,manifest['size']):raise ValueError('Stale LUT model')
            rows=scope_rows(node,model)
            if manifest['rows']!=[list(map(float,row)) for matrix in rows for row in matrix]:
                raise ValueError('Stale LUT endpoints')
        return manifest


    def verify_replacement(node, expected):
        nodes = _application.component('graph.nodes')
        adopt_native(node, nodes.output_intent(node))


    def _graph_snapshot(node):
        """Rollback values only: publication never changes native references."""
        spaces = _application.component('graph.spaces')
        return dict(selector=resources.selected_index(node),
            rows=spaces.read_rows(node),
            manifest=node.get(MANIFEST),pending=node.get(INCOMPLETE))


    def _restore_graph(node,snapshot):
        _fallback(node)
        _write_rows(node,snapshot['rows'])
        for name,key in ((MANIFEST,'manifest'),(INCOMPLETE,'pending')):
            if snapshot[key] is None:
                if name in node:del node[name]
            else:node[name]=snapshot[key]
        index=snapshot['selector']
        resources.select_payload(node,index)
        if resources.selected_index(node)!=index:raise RuntimeError('Rollback selector differs')


    def _rollback_graph(node,snapshot):
        current=resources.record(node.node_tree,node);token=current.get('graph_token')
        try:
            _restore_graph(node,snapshot)
        except Exception:
            current['repair_selection']=snapshot
            node[INCOMPLETE]=True
            if token is not None:resources.graph_receipt(node,'graph_fail',token)
            log_exception(LOG,'Perceptual Grade LUT rollback requires recovery')
            raise
        else:
            if token is not None:resources.graph_receipt(node,'graph_fail',token,validated=True)
            current.pop('graph_token',None)


    def _repair_graph_native(node):
        """Retry compensation for an unfinished current-generation publication."""
        output_guard = _application.component('graph.output_guard')
        if not output_guard.publishing():raise RuntimeError('Graph repair requires a native resource lease')
        current=resources.record(node.node_tree,node)
        repair=current.get('repair_selection')
        if repair is not None:
            _restore_graph(node,repair)
            token=current.get('graph_token')
            if token is not None:resources.graph_receipt(node,'graph_repair',token,validated=True)
            current.pop('repair_selection',None);current.pop('graph_token',None)


    def _repair_graph(node):
        if is_adopted(node):return
        current=resources.record(node.node_tree,node)
        protocol=current.get('publication_protocol')
        repairing=protocol is not None and protocol.state.debt
        if repairing:
            protocol.event('failed')
            protocol.require('repair','repair')
        try:
            if current.get('publication_repair') is not None:
                _repair_publication_native(node,current)
            else:
                _repair_graph_native(node)
        except Exception:
            if repairing:protocol.event('failed')
            raise
        if repairing:
            protocol.require('repaired','release',readable=True,validated=True)
        current.pop('publication_repair',None)
        current.pop('repair_rebind',None)


    def _repair_publication_native(node,current):
        """Replay the whole rollback from values; no RNA survives a failed call."""
        repair=current['publication_repair']
        image=resources.bank_images(node)[repair['index']]
        if int(image.session_uid)!=repair['image_uid']:
            raise RuntimeError('LUT rollback bank changed without native replacement')
        reservation=repair['reservation']
        if reservation is not None and current['state'].pending==reservation['index']:
            resources.receipt(node,'fail',reservation)
        try:
            if current.get('graph_token') is not None:
                _rollback_graph(node,repair['snapshot'])
                current.pop('repair_selection',None)
            elif repair['graph_started']:
                _restore_graph(node,repair['snapshot'])
                current.pop('repair_selection',None)
        finally:
            _cubes.pop(image.as_pointer(),None)


    def _publication_checkpoint(phase):
        """Fault-injection seam for actual native transaction boundaries."""


    def installed_matches(node,model):
        image=image_for(node)
        return image is not None and image.get('_pg_cube_schema')==SCHEMA and image.get('_pg_cube_signature') in (
            cm.signature(model),cm.signature(model,cm.PREVIEW_SIZE))


    def backend_status(node,backend,device='',reason=None):
        values={'_pg_cube_backend':backend,'_pg_cube_device':device,
                '_pg_cube_backend_reason':reason or ''}
        for key,value in values.items():
            if node.get(key)!=value:node[key]=value


    def draw_error(layout,node):
        """Only actionable failures belong in the node and sidebar UI."""
        configuration=node.get('_pg_error');error=node.get('_pg_bake_error')
        if not configuration and not error:return
        box=layout.box();box.alert=True
        if configuration:
            box.label(text=tr('error.grade_bypassed',error=error_text(configuration)),icon='ERROR')
        else:
            if node.get('_pg_bake_render_error'):
                box.label(text=tr('bake.failed_render'),icon='ERROR')
            box.label(text=tr('bake.error',error=error_text(error)),icon='ERROR')


    def clear_error(node):
        """A pending retry does not clear a failure; a current successful LUT does."""
        for key in ('_pg_bake_error','_pg_bake_render_error'):
            if key in node:del node[key]


    def publish(node,baker,force=False,preview=False):
        output_guard = _application.component('graph.output_guard'); compositor = _application.component('graph.compositor')
        if not output_guard.available():return None
        if not force and not compositor.can_commit(node):return None
        # Scope capture may evaluate Blender data. Finish it before acquiring the
        # non-reentrant native payload lease, whose entry rechecks job admission.
        jobs = _application.component('scopes.jobs')
        jobs.linear_tick()
        cube=baker.preview if preview else baker.cube
        if cube is None:return False
        prepared_pixels = _application.component('jobs.baker').prepared_pixels
        pixels=prepared_pixels(cube,node.node_tree.bl_idname=='CompositorNodeTree',wait=force)
        if pixels is None:return None
        if not preview and hasattr(baker,'prepare_publication') and not baker.prepare_publication():return False
        try:
            with output_guard.publication():
                published=_publish(node,baker,force,preview,pixels)
            if published and not preview and hasattr(baker,'published') and not baker.published():
                raise RuntimeError('LUT workflow rejected publication receipt')
            return published
        except output_guard.PublicationDeferred:
            return None
        except Exception:
            if not preview and hasattr(baker,'publication_failed'):baker.publication_failed()
            raise


    def _publish(node,baker,force,preview,pixels):
        if is_adopted(node):return False
        if not scheduler.lifecycle_admits('publish'):return False
        if not uses_lut(node):return False
        # An active immutable 33-cube must be allowed to finish during continuous
        # edits. Final cubes always describe the current controls.
        nodes = _application.component('graph.nodes')
        current=nodes.bake_model(node)
        current_key=cm.signature(current)
        state=_states.get(node.as_pointer())
        active=(state is not None and state['baker'] is baker and not state['preview_published']
                and state['context']==context(node))
        if baker.signature!=current_key and not (preview and active and scheduler.dragging(node)):return False
        compositor = _application.component('graph.compositor')
        ticket=scheduler.lut_candidate(node,baker.model,preview)
        scheduler.ready(ticket,True)
        if not scheduler.publication(ticket,commit=False):
            # Backend fallback may recreate an already installed current preview.
            # Acknowledge it without another image/graph/scope publication.
            image=image_for(node)
            return bool(image is not None and image.get('_pg_cube_signature')==
                        cm.signature(current,cm.PREVIEW_SIZE if preview else cm.SIZE))
        cube=baker.preview if preview else baker.cube
        old=image_for(node)
        # A delayed preview cannot demote an already installed final cube.
        if (preview and not node.get(INCOMPLETE) and old is not None
                and old.get('_pg_cube_schema')==SCHEMA and old.get('_pg_cube_signature')==current_key):return False
        controller=resources.record(node.node_tree,node)
        _repair_graph(node)
        snapshot=_graph_snapshot(node)
        images=resources.bank_images(node)
        native_selected=resources.selected_index(node)
        index=next(index for index in ((0,1) if cube.size==33 else (2,3))
                   if index not in (controller['state'].selected,native_selected))
        image=images[index]
        report=dict(cube.diagnostics);report.setdefault('backend',baker.backend)
        coordinator=controller.get('publication_protocol')
        if coordinator is None:
            coordinator=controller['publication_protocol']=Protocol(_application,'publication')
        coordinator.identity.update(operation=next(_publication_serial), generation=1,
                                    resource=int(image.session_uid), target=int(node.as_pointer()))
        coordinator.require('begin','reserve')
        reservation=None
        try:
            reservation=resources.reserve(node,cube.size,cube.signature)
            if not preview and hasattr(baker,'authorize_publication'):baker.authorize_publication()
        except Exception:
            if reservation is not None:resources.receipt(node,'fail',reservation)
            coordinator.require('aborted','release',validated=True)
            raise
        repair=dict(index=index,image_uid=int(image.session_uid),reservation=reservation,
                    snapshot=snapshot,mutated=False,graph_started=False)
        controller['publication_repair']=repair
        def rollback():
            if not repair['mutated']:
                if reservation is not None and controller['state'].pending==index:
                    resources.receipt(node,'fail',reservation)
                coordinator.require('aborted','release',validated=True)
                controller.pop('publication_repair',None)
                return
            coordinator.event('failed')
            coordinator.require('repair','repair')
            try:
                _repair_publication_native(node,controller)
            except Exception:
                coordinator.event('failed')
                raise
            coordinator.require('repaired','release',readable=True,validated=True)
            controller.pop('publication_repair',None)
        def install():
            coordinator.require('reserved','write',accepted=True)
            _publication_checkpoint('reserved')
            repair['mutated']=True
            resources.arm(node,reservation,image)
            _publication_checkpoint('armed')
            resources.transfer_payload(image,pixels,cube,report,node=node,reservation=reservation)
            _publication_checkpoint('transferred')
            resources.receipt(node,'transferred',reservation,validated=True)
            coordinator.require('written','verify')
            coordinator.require('verified','publish',readable=True,validated=True)
            _publication_checkpoint('verified')
            rows=scope_rows(node,current if baker.signature==current_key else baker.model)
            repair['graph_started']=True
            manifest=_install_graph(node,index,_checked_rows(*rows),checkpoints=True,defer=True)
            if manifest['signature']!=cube.signature:raise ValueError('LUT signature changed during install')
            return image
        try:
            scheduler.effect(ticket,'install',install,rollback=rollback)
        except Exception:
            # effect_begin can reject before invoking either callback. A failed
            # rollback, in contrast, retains its complete recipe for recovery.
            if coordinator.state.debt and coordinator.state.phase in ('reserving','writing'):
                rollback()
            raise
        # Both reducers commit before returning the host lease to native readers.
        if not scheduler.publication(ticket):
            rollback()
            raise RuntimeError('LUT installation lost its scheduling lease')
        resources.graph_receipt(node,'select',controller['graph_token'],validated=True)
        controller.pop('graph_token',None)
        if not coordinator.event('published',readable=True,validated=True).accepted:
            raise RuntimeError('LUT publication workflow rejected its native receipt')
        controller.pop('publication_repair',None)
        _cubes[image.as_pointer()]=(retain(image,'lut.cube_image'),cube,image.get(resources.REVISION,0))
        if not preview:
            state=_states.get(node.as_pointer())
            if state is not None and state['baker'] is baker:state['final_published']=True
            _remember_final(cube)
        # Notifications enqueue native work; none may enter evaluation recursively.
        try:
            _publication_checkpoint('cleanup')
            backend_status(node,report['backend'],report.get('device',''),getattr(baker,'fallback_reason',None))
            scheduler._updates.pop(node.as_pointer(),None)
            if baker.signature==current_key:clear_error(node)
            compositor.tag_output(node)
            if preview:compositor.preview_installed(node)
            redraw()
        except Exception:log_exception(LOG,'Perceptual Grade postcommit notification failed')
        return True


    def redraw():
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='NODE_EDITOR':area.tag_redraw()


    def cpu_baker(model,progressive,encoded,stack):
        RemoteBaker = _application.component('jobs.baker').RemoteBaker
        return RemoteBaker(model,progressive,encoded,stack)


    def acquire_gpu(state):
        gpu = _application.component('graph.gpu')
        baker=state['baker']
        if state['stack']:return gpu.acquire(baker.model,baker.progressive,state['encoded'],stack=True)
        return gpu.acquire(baker.model,baker.progressive,state['encoded'])


    def new_state(node,model):
        nodes = _application.component('graph.nodes')
        now=time.monotonic();progressive=baked_only(node)
        regular=nodes.is_grade(node);encoded=node.node_tree.bl_idname=='CompositorNodeTree'
        baker=cpu_baker(model,progressive,encoded,not regular)
        return {'node':retain(node,'lut.state'),'tree':node.id_data.as_pointer(),'baker':baker,'auto':regular or bool(model.entries),'acquired':False,
                'stack':not regular,
                'encoded':encoded,
                'due':now+(PREVIEW_DELAY if progressive else .25),
                'future':None,'preview_published':False,'final_published':False,
                'context':context(node),'started':False,'pending':None,'superseded':False,
                'schedule':scheduler.lut_request(node,model)}


    def request(node,model,force=False):
        if is_adopted(node) or not uses_lut(node):return
        scheduler.lut_request(node,model)
        controller=resources.record(node.node_tree,node)
        if controller.get('repair_selection') is not None:
            if not scheduler.lifecycle_admits('publish'):return
            output_guard = _application.component('graph.output_guard')
            if output_guard.publishing():_repair_graph(node)
            elif output_guard.available():
                with output_guard.publication():_repair_graph(node)
            else:return
        previous=_states.get(node.as_pointer())
        if previous is not None and previous['schedule'].session!=scheduler.session():
            stop(_states.pop(node.as_pointer(),None))
        key=cm.signature(model);image=image_for(node)
        failure_key=(node.as_pointer(),context(node),key)
        if not force and failure_key in _failures:return
        if force:_failures.pop(failure_key,None)
        valid_image=(image is not None and image.get('_pg_cube_schema')==SCHEMA
            and not node.get(INCOMPLETE) and resources.policy.complete(controller['state'],
                resources.selected_index(node),image.get('_pg_cube_signature')))
        if valid_image and image.get('_pg_cube_signature')==key:
            scheduler.lut_cache_hit(node,model)
            state=_states.get(node.as_pointer())
            if state is not None and (state['baker'].signature!=key or state.get('future') is not None):
                stop(_states.pop(node.as_pointer(),None))
            compositor = _application.component('graph.compositor')
            if compositor.can_commit(node):select(node,True)
            clear_error(node)
            backend_status(node,image.get('_pg_bake_backend','CPU'),image.get('_pg_bake_device',''))
            return
        state=_states.get(node.as_pointer())
        if state is not None and not force and state['context']==context(node) and key==state['baker'].signature:
            state['pending']=model if state['superseded'] else None
            if state['superseded']:scheduler.wake(tick,PREVIEW_DELAY)
            return
        # Keep the real-time result while a current cube is prepared, including
        # when a packed cube from previous settings still exists.
        compositor = _application.component('graph.compositor')
        if compositor.can_commit(node):select(node,valid_image if baked_only(node) else False)
        pointer=node.as_pointer();state=_states.get(pointer)
        decision=(scheduler.replacement(node,state['baker'].signature,state['started'],
            state['baker'].preview is not None,finished=_finished(state),published=state['preview_published'],
            cancelled=state['superseded'],scope_demand=_scope_protected(state))
            if state is not None else None)
        if state is None:
            state=new_state(node,model)
            _states[pointer]=state
        elif (force or state['context']!=context(node) or state['final_published']
              or decision.replace):
            due=state['due']
            stop(state)
            state=new_state(node,model)
            # Replacing unfinished work must not perpetually restart the first
            # launch interval. The installed preview remains visible meanwhile.
            state['due']=min(due,state['due'])
            _states[pointer]=state
        elif state['baker'].signature!=key or state['superseded']:
            state['pending']=model
            if decision.cancel:
                stop(state);state['superseded']=True
        _states.move_to_end(pointer)
        # Bound analytic coordinate caches. Packed LUTs remain attached to grades.
        for old,state_old in list(_states.items())[:-2]:
            if state_old['final_published'] and state_old.get('future') is None:_states.pop(old,None)
        if key in _ready:
            # Never mutate a Baker that a worker is still evaluating. In
            # particular, a final cache hit must supersede its preview job.
            if state.get('future') is not None or state['baker'].signature!=key or state['superseded']:
                stop(state);state=new_state(node,model);_states[pointer]=state
            state['baker'].cube=_ready[key]
            if publish(node,state['baker'],force=force):return
        scheduler.wake(tick,PREVIEW_DELAY)


    def settle(node):
        """Release fences an old protected preview without waiting for its worker."""
        nodes = _application.component('graph.nodes')
        if uses_lut(node):
            request(node,nodes.bake_model(node))


    def live_nodes():
        """Resolve queued pointers from live owners before touching stored RNA.

        A removed custom node can retain a nonzero pointer without safe access to
        its properties. Do not rely on as_pointer() to detect deletion.
        """
        owners={state['tree'] for state in _states.values()}
        trees=list(bpy.data.node_groups)
        for collection in (bpy.data.materials,bpy.data.worlds,bpy.data.lights):
            trees.extend(item.node_tree for item in collection if item.node_tree)
        return {node.as_pointer():node for tree in trees if tree.as_pointer() in owners
                for node in tree.nodes if node.as_pointer() in _states}


    def tick():
        if not scheduler.lifecycle_admits('start'):return .016
        get_runtime = _application.component('services').get_runtime
        get_runtime().poll()
        jobs = _application.component('scopes.jobs')
        jobs.linear_tick()
        scope_tick()
        collect_retired()
        pending=False;live=live_nodes()
        ordered=sorted(tuple(_states.items()),key=lambda item:item[1]['preview_published'])
        for pointer,state in ordered:
            try:
                node=live.get(pointer);baker=state['baker']
                if node is None:stop(_states.pop(pointer,None));continue
                state['node']=retain(node,'lut.state')
                if not state['started'] and baker.cube is None and baker.signature in _ready:
                    # A frozen scope demand may have finished while this native
                    # preview waited for the same CPU lane. Reuse its exact cube.
                    baker.cube=_ready[baker.signature]
                future=state.get('future')
                if future is not None:
                    pending=True
                    if future.done():
                        try:scheduler.effect(state['schedule'],'worker',future.result)
                        except CancelledError:
                            if not state['superseded']:raise
                        state['future']=None
                if not state['superseded'] and hasattr(baker,'collect'):baker.collect()
                if state['future'] is None:_remember_final(baker.cube)
                if state['pending'] is not None:
                    decision=scheduler.replacement(node,baker.signature,state['started'],
                        baker.preview is not None,finished=_finished(state),
                        published=state['preview_published'],cancelled=state['superseded'],scope_demand=_scope_protected(state))
                    if decision.cancel and not state['superseded']:
                        stop(state);state['superseded']=True
                    if decision.replace:
                        model=state['pending'];stop(state)
                        state=new_state(node,model);state['due']=time.monotonic()
                        _states[pointer]=state;pending=True
                        continue
                # One worker calculates the nested preview first, then continues
                # the remaining vertices without waiting for timer/publication.
                if (not state['superseded'] and baker.cube is None and time.monotonic()>=state['due']
                        and not _retired and _scope_running is None and not any(s.get('future') is not None for s in _states.values())):
                    if not state['started'] and not scheduler.scope_start(state['schedule']):
                        pending=True;continue
                    state['started']=True
                    if state['auto'] and not state['acquired']:
                        accelerated,reason=scheduler.effect(state['schedule'],'gpu',lambda:acquire_gpu(state))
                        if accelerated is not None:baker=accelerated;state['baker']=baker
                        else:baker.fallback_reason=reason
                        state['acquired']=True
                    backend_status(node,baker.backend,getattr(baker,'device',''),getattr(baker,'fallback_reason',None))
                    state['future']=scheduler.effect(state['schedule'],'submit',baker.start)
                if baker.preview is not None and not state['preview_published']:
                    # Release revokes an obsolete preview that was not accepted
                    # while its gesture was active. Keep its pending successor
                    # queued until this physical worker retires instead of
                    # discarding the successor along with the stale snapshot.
                    if state['pending'] is not None and not scheduler.dragging(node):
                        pending=True
                        continue
                    result=publish(node,baker,preview=True)
                    if result is None:pending=True;continue
                    if result is False:
                        if state['pending'] is not None:
                            pending=True
                        else:
                            stop(_states.pop(pointer,None))
                        continue
                    state['preview_published']=True
                    # Let Blender dispatch the preview frame before replacing
                    # its image, even if refinement already finished in parallel.
                    pending=True;continue
                if baker.cube is not None and state['future'] is None:
                    _remember_final(baker.cube)
                    scheduler.scope_complete(state['schedule'])
                    image=image_for(node)
                    if image is None or node.get(INCOMPLETE) or image.get('_pg_cube_signature')!=baker.signature:
                        result=publish(node,baker)
                        if result is None:pending=True
                        elif result is False:stop(_states.pop(pointer,None))
                    continue
                pending=True
            except CancelledError:
                if _states.get(pointer) is state:_states.pop(pointer,None)
            except Exception as exc:
                log_exception(LOG,'Perceptual Grade LUT bake failed')
                _failures[(node.as_pointer(),state['context'],baker.signature)]=diagnostic_text(exc)
                if baker.signature in _scope_demands:
                    _scope_failures[baker.signature]=diagnostic_text(exc)
                    _scope_demands.pop(baker.signature,None)
                stop(state)
                try:
                    nodes = _application.component('graph.nodes')
                    if (state['context']==context(node)
                            and state['baker'].signature==cm.signature(nodes.bake_model(node))):
                        node['_pg_bake_error']=diagnostic_text(exc)
                        if '_pg_bake_render_error' in node:del node['_pg_bake_render_error']
                        redraw()
                except (ValueError,ArithmeticError,ReferenceError,RuntimeError):pass
                if state['pending'] is not None:
                    replacement=new_state(node,state['pending']);replacement['due']=time.monotonic()
                    _states[pointer]=replacement;pending=True
                else:_states.pop(pointer,None)
        return .016 if pending else None


    def ensure(node,model,force=False):
        """Rendering waits for the exact current signature, never an old grade."""
        try:return _ensure(node,model,force)
        except Exception as exc:
            key=cm.signature(model)
            _failures[(node.as_pointer(),context(node),key)]=diagnostic_text(exc)
            if key in _scope_demands:
                _scope_failures[key]=diagnostic_text(exc);_scope_demands.pop(key,None)
            stop(_states.pop(node.as_pointer(),None))
            node['_pg_bake_error']=diagnostic_text(exc)
            raise


    def _ensure(node,model,force):
        if is_adopted(node):return
        compositor = _application.component('graph.compositor')
        compositor.flush_update(node)
        if not uses_lut(node):return
        # A synchronous save/render shares the physical lane with a detached
        # final scope bake. Retire it and consume its cache before requesting a
        # native bake, including when the exact same model is requested here.
        if _scope_running is not None:
            try:_scope_running[2].result()
            except Exception:pass  # scope_tick records the failure receipt below.
            scope_tick(start=False)
        request(node,model,force)
        image=image_for(node)
        controller=resources.record(node.node_tree,node)
        if (image is not None and not node.get(INCOMPLETE) and image.get('_pg_cube_schema')==SCHEMA
                and resources.policy.complete(controller['state'],resources.selected_index(node),cm.signature(model))):return
        state=_states.get(node.as_pointer())
        if state is None:
            raise RuntimeError(_failures.get((node.as_pointer(),context(node),cm.signature(model)),
                'No current LUT bake is available; retry the failed request'))
        if state['baker'].signature!=cm.signature(model) or state['superseded']:
            stop(state);state=new_state(node,model);_states[node.as_pointer()]=state
        # Final rendering is synchronous; retire obsolete CPU work before starting
        # the exact requested job, rather than growing a hidden executor queue.
        for future in tuple(_retired):
            try:future.result()
            except Exception:pass
            _retired.discard(future)
        collect_retired()
        for other in tuple(_states.values()):
            if other is state or other.get('future') is None:continue
            try:other['future'].result()
            except Exception:pass  # Its timer owns the failure receipt.
        if state.get('future') is not None:
            try:scheduler.effect(state['schedule'],'worker',state['future'].result)
            finally:state['future']=None
        if state['baker'].signature!=cm.signature(model):
            stop(state);state=new_state(node,model);_states[node.as_pointer()]=state
        if state['auto'] and not state['acquired'] and state['baker'].cube is None:
            accelerated,reason=scheduler.effect(state['schedule'],'gpu',lambda:acquire_gpu(state))
            if accelerated is not None:state['baker']=accelerated
            else:state['baker'].fallback_reason=reason
            state['acquired']=True
        baker=state['baker']
        scheduler.effect(state['schedule'],'worker',baker.finish)
        prepared_pixels = _application.component('jobs.baker').prepared_pixels
        prepared_pixels(baker.cube,state['encoded'],wait=True)
        backend_status(node,baker.backend,getattr(baker,'device',''),getattr(baker,'fallback_reason',None))
        scheduler.scope_complete(state['schedule'])
        if not publish(node,state['baker'],force):raise RuntimeError('Grade changed while finishing its LUT')


    def clear(release_gpu=False):
        nonlocal _scope_running
        scheduler.unwake(tick)
        scheduler.unwake(scope_tick)
        if _scope_running is not None:
            _,baker,future,ticket=_scope_running;baker.cancel();future.cancel()
            if not _physically_done(future):
                _retired.add(future)
                _retirements[ticket]=(future,)
            else:scheduler.scope_complete(ticket)
            _scope_running=None
        _scope_demands.clear();_scope_failures.clear();_failures.clear()
        scheduler.clear()
        for state in _states.values():
            stop(state)
        _states.clear();_cubes.clear()
        from ..model import baking
        if release_gpu:
            resources.clear()
            _ready.clear()
            gpu = _application.component('graph.gpu')
            gpu.clear()
        baking.clear()

    def _set_export_has_runtime_intent(value):
        nonlocal has_runtime_intent
        has_runtime_intent = value

    def _set_export_adopt_native(value):
        nonlocal adopt_native
        adopt_native = value

    def _set_export_is_adopted(value):
        nonlocal is_adopted
        is_adopted = value

    def _set_export_request_intent(value):
        nonlocal request_intent
        request_intent = value

    def _set_export_INCOMPLETE(value):
        nonlocal INCOMPLETE
        INCOMPLETE = value

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export_MANIFEST(value):
        nonlocal MANIFEST
        MANIFEST = value

    def _set_export_PREVIEW_DELAY(value):
        nonlocal PREVIEW_DELAY
        PREVIEW_DELAY = value

    def _set_export_ROUTE(value):
        nonlocal ROUTE
        ROUTE = value

    def _set_export_SCHEMA(value):
        nonlocal SCHEMA
        SCHEMA = value

    def _set_export__cubes(value):
        nonlocal _cubes
        _cubes = value

    def _set_export__failures(value):
        nonlocal _failures
        _failures = value

    def _set_export__ready(value):
        nonlocal _ready
        _ready = value

    def _set_export__retired(value):
        nonlocal _retired
        _retired = value

    def _set_export__retirements(value):
        nonlocal _retirements
        _retirements = value

    def _set_export__scope_demands(value):
        nonlocal _scope_demands
        _scope_demands = value

    def _set_export__scope_failures(value):
        nonlocal _scope_failures
        _scope_failures = value

    def _set_export__scope_running(value):
        nonlocal _scope_running
        _scope_running = value

    def _set_export__states(value):
        nonlocal _states
        _states = value

    def _set_export_ScopePreparing(value):
        nonlocal ScopePreparing
        ScopePreparing = value

    def _set_export_ScopeFailure(value):
        nonlocal ScopeFailure
        ScopeFailure = value

    def _set_export__Graph(value):
        nonlocal _Graph
        _Graph = value

    def _set_export_Graph(value):
        nonlocal Graph
        Graph = value

    def _set_export_compositor_helper(value):
        nonlocal compositor_helper
        compositor_helper = value

    def _set_export_compositor_lookup(value):
        nonlocal compositor_lookup
        compositor_lookup = value

    def _set_export_prepare_helpers(value):
        nonlocal prepare_helpers
        prepare_helpers = value

    def _set_export_shader_lookup(value):
        nonlocal shader_lookup
        shader_lookup = value

    def _set_export_repair_input(value):
        nonlocal repair_input
        repair_input = value

    def _set_export__make_bank(value):
        nonlocal _make_bank
        _make_bank = value

    def _set_export_clone_kernel(value):
        nonlocal clone_kernel
        clone_kernel = value

    def _set_export_make_kernel(value):
        nonlocal make_kernel
        make_kernel = value

    def _set_export__build_kernel(value):
        nonlocal _build_kernel
        _build_kernel = value

    def _set_export_select(value):
        nonlocal select
        select = value

    def _set_export_image_for(value):
        nonlocal image_for
        image_for = value

    def _set_export_read_cube(value):
        nonlocal read_cube
        read_cube = value

    def _set_export_installed_cube(value):
        nonlocal installed_cube
        installed_cube = value

    def _set_export_installed_signal(value):
        nonlocal installed_signal
        installed_signal = value

    def _set_export_signal_signature(value):
        nonlocal signal_signature
        signal_signature = value

    def _set_export_scope_rows(value):
        nonlocal scope_rows
        scope_rows = value

    def _set_export_scope_signature(value):
        nonlocal scope_signature
        scope_signature = value

    def _set_export__remember_final(value):
        nonlocal _remember_final
        _remember_final = value

    def _set_export__scope_owner(value):
        nonlocal _scope_owner
        _scope_owner = value

    def _set_export__scope_drop_owner(value):
        nonlocal _scope_drop_owner
        _scope_drop_owner = value

    def _set_export_scope_signal(value):
        nonlocal scope_signal
        scope_signal = value

    def _set_export_scope_release(value):
        nonlocal scope_release
        scope_release = value

    def _set_export_scope_retry(value):
        nonlocal scope_retry
        scope_retry = value

    def _set_export__scope_protected(value):
        nonlocal _scope_protected
        _scope_protected = value

    def _set_export_scope_tick(value):
        nonlocal scope_tick
        scope_tick = value

    def _set_export_evaluate_signal(value):
        nonlocal evaluate_signal
        evaluate_signal = value

    def _set_export_uses_lut(value):
        nonlocal uses_lut
        uses_lut = value

    def _set_export_baked_only(value):
        nonlocal baked_only
        baked_only = value

    def _set_export_stop(value):
        nonlocal stop
        stop = value

    def _set_export_collect_retired(value):
        nonlocal collect_retired
        collect_retired = value

    def _set_export__finished(value):
        nonlocal _finished
        _finished = value

    def _set_export_context(value):
        nonlocal context
        context = value

    def _set_export_commit_rows(value):
        nonlocal commit_rows
        commit_rows = value

    def _set_export__checked_rows(value):
        nonlocal _checked_rows
        _checked_rows = value

    def _set_export__write_rows(value):
        nonlocal _write_rows
        _write_rows = value

    def _set_export__output_key(value):
        nonlocal _output_key
        _output_key = value

    def _set_export__fallback(value):
        nonlocal _fallback
        _fallback = value

    def _set_export__install_graph(value):
        nonlocal _install_graph
        _install_graph = value

    def _set_export_commit_endpoint_rows(value):
        nonlocal commit_endpoint_rows
        commit_endpoint_rows = value

    def _set_export_refresh_manifest(value):
        nonlocal refresh_manifest
        refresh_manifest = value

    def _set_export__output_snapshot(value):
        nonlocal _output_snapshot
        _output_snapshot = value

    def _set_export__manifest(value):
        nonlocal _manifest
        _manifest = value

    def _set_export_validate_output(value):
        nonlocal validate_output
        validate_output = value

    def _set_export_verify_replacement(value):
        nonlocal verify_replacement
        verify_replacement = value

    def _set_export__graph_snapshot(value):
        nonlocal _graph_snapshot
        _graph_snapshot = value

    def _set_export__restore_graph(value):
        nonlocal _restore_graph
        _restore_graph = value

    def _set_export__rollback_graph(value):
        nonlocal _rollback_graph
        _rollback_graph = value

    def _set_export__repair_graph(value):
        nonlocal _repair_graph
        _repair_graph = value

    def _set_export__publication_checkpoint(value):
        nonlocal _publication_checkpoint
        _publication_checkpoint = value

    def _set_export_installed_matches(value):
        nonlocal installed_matches
        installed_matches = value

    def _set_export_backend_status(value):
        nonlocal backend_status
        backend_status = value

    def _set_export_draw_error(value):
        nonlocal draw_error
        draw_error = value

    def _set_export_clear_error(value):
        nonlocal clear_error
        clear_error = value

    def _set_export_publish(value):
        nonlocal publish
        publish = value

    def _set_export__publish(value):
        nonlocal _publish
        _publish = value

    def _set_export_redraw(value):
        nonlocal redraw
        redraw = value

    def _set_export_cpu_baker(value):
        nonlocal cpu_baker
        cpu_baker = value

    def _set_export_acquire_gpu(value):
        nonlocal acquire_gpu
        acquire_gpu = value

    def _set_export_new_state(value):
        nonlocal new_state
        new_state = value

    def _set_export_request(value):
        nonlocal request
        request = value

    def _set_export_settle(value):
        nonlocal settle
        settle = value

    def _set_export_live_nodes(value):
        nonlocal live_nodes
        live_nodes = value

    def _set_export_tick(value):
        nonlocal tick
        tick = value

    def _set_export_ensure(value):
        nonlocal ensure
        ensure = value

    def _set_export__ensure(value):
        nonlocal _ensure
        _ensure = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CancelledError': lambda: CancelledError,
        'Graph': lambda: Graph,
        'INCOMPLETE': lambda: INCOMPLETE,
        'LOG': lambda: LOG,
        'MANIFEST': lambda: MANIFEST,
        'OrderedDict': lambda: OrderedDict,
        'PREVIEW_DELAY': lambda: PREVIEW_DELAY,
        'ROUTE': lambda: ROUTE,
        'adopt_native': lambda: adopt_native,
        'is_adopted': lambda: is_adopted,
        'has_runtime_intent': lambda: has_runtime_intent,
        'request_intent': lambda: request_intent,
        'SCHEMA': lambda: SCHEMA,
        'ScopeFailure': lambda: ScopeFailure,
        'ScopePreparing': lambda: ScopePreparing,
        '_Graph': lambda: _Graph,
        '_build_kernel': lambda: _build_kernel,
        '_checked_rows': lambda: _checked_rows,
        '_cubes': lambda: _cubes,
        '_ensure': lambda: _ensure,
        '_failures': lambda: _failures,
        '_fallback': lambda: _fallback,
        '_finished': lambda: _finished,
        '_graph_snapshot': lambda: _graph_snapshot,
        '_install_graph': lambda: _install_graph,
        '_make_bank': lambda: _make_bank,
        '_manifest': lambda: _manifest,
        '_output_key': lambda: _output_key,
        '_output_snapshot': lambda: _output_snapshot,
        '_publication_checkpoint': lambda: _publication_checkpoint,
        '_publish': lambda: _publish,
        '_ready': lambda: _ready,
        '_remember_final': lambda: _remember_final,
        '_repair_graph': lambda: _repair_graph,
        '_restore_graph': lambda: _restore_graph,
        '_retired': lambda: _retired,
        '_retirements': lambda: _retirements,
        '_rollback_graph': lambda: _rollback_graph,
        '_scope_demands': lambda: _scope_demands,
        '_scope_drop_owner': lambda: _scope_drop_owner,
        '_scope_failures': lambda: _scope_failures,
        '_scope_owner': lambda: _scope_owner,
        '_scope_protected': lambda: _scope_protected,
        '_scope_running': lambda: _scope_running,
        '_states': lambda: _states,
        '_write_rows': lambda: _write_rows,
        'acquire_gpu': lambda: acquire_gpu,
        'backend_status': lambda: backend_status,
        'baked_only': lambda: baked_only,
        'bpy': lambda: bpy,
        'clear': lambda: clear,
        'clear_error': lambda: clear_error,
        'clone_kernel': lambda: clone_kernel,
        'cm': lambda: cm,
        'collect_retired': lambda: collect_retired,
        'commit_endpoint_rows': lambda: commit_endpoint_rows,
        'commit_rows': lambda: commit_rows,
        'compositor_helper': lambda: compositor_helper,
        'compositor_lookup': lambda: compositor_lookup,
        'context': lambda: context,
        'cpu_baker': lambda: cpu_baker,
        'diagnostic_text': lambda: diagnostic_text,
        'draw_error': lambda: draw_error,
        'ensure': lambda: ensure,
        'error_text': lambda: error_text,
        'evaluate_signal': lambda: evaluate_signal,
        'image_for': lambda: image_for,
        'installed_cube': lambda: installed_cube,
        'installed_matches': lambda: installed_matches,
        'installed_signal': lambda: installed_signal,
        'json': lambda: json,
        'live_nodes': lambda: live_nodes,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'make_kernel': lambda: make_kernel,
        'math': lambda: math,
        'new_state': lambda: new_state,
        'np': lambda: np,
        'prepare_helpers': lambda: prepare_helpers,
        'publish': lambda: publish,
        'read_cube': lambda: read_cube,
        'redraw': lambda: redraw,
        'refresh_manifest': lambda: refresh_manifest,
        'remove_image': lambda: remove_image,
        'repair_input': lambda: repair_input,
        'request': lambda: request,
        'resolve': lambda: resolve,
        'resources': lambda: resources,
        'retain': lambda: retain,
        'scheduler': lambda: scheduler,
        'scope_release': lambda: scope_release,
        'scope_retry': lambda: scope_retry,
        'scope_rows': lambda: scope_rows,
        'scope_signal': lambda: scope_signal,
        'scope_signature': lambda: scope_signature,
        'scope_tick': lambda: scope_tick,
        'select': lambda: select,
        'settle': lambda: settle,
        'shader_lookup': lambda: shader_lookup,
        'signal_signature': lambda: signal_signature,
        'stop': lambda: stop,
        'tick': lambda: tick,
        'time': lambda: time,
        'tr': lambda: tr,
        'uses_lut': lambda: uses_lut,
        'validate_output': lambda: validate_output,
        'verify_replacement': lambda: verify_replacement,
    }, writers={'has_runtime_intent': _set_export_has_runtime_intent, 'adopt_native': _set_export_adopt_native, 'is_adopted': _set_export_is_adopted, 'request_intent': _set_export_request_intent, 
        'INCOMPLETE': _set_export_INCOMPLETE,
        'LOG': _set_export_LOG,
        'MANIFEST': _set_export_MANIFEST,
        'PREVIEW_DELAY': _set_export_PREVIEW_DELAY,
        'ROUTE': _set_export_ROUTE,
        'SCHEMA': _set_export_SCHEMA,
        '_cubes': _set_export__cubes,
        '_failures': _set_export__failures,
        '_ready': _set_export__ready,
        '_retired': _set_export__retired,
        '_retirements': _set_export__retirements,
        '_scope_demands': _set_export__scope_demands,
        '_scope_failures': _set_export__scope_failures,
        '_scope_running': _set_export__scope_running,
        '_states': _set_export__states,
    'ScopePreparing': _set_export_ScopePreparing, 'ScopeFailure': _set_export_ScopeFailure, '_Graph': _set_export__Graph, 'Graph': _set_export_Graph, 'compositor_helper': _set_export_compositor_helper, 'compositor_lookup': _set_export_compositor_lookup, 'prepare_helpers': _set_export_prepare_helpers, 'shader_lookup': _set_export_shader_lookup, 'repair_input': _set_export_repair_input, '_make_bank': _set_export__make_bank, 'clone_kernel': _set_export_clone_kernel, 'make_kernel': _set_export_make_kernel, '_build_kernel': _set_export__build_kernel, 'select': _set_export_select, 'image_for': _set_export_image_for, 'read_cube': _set_export_read_cube, 'installed_cube': _set_export_installed_cube, 'installed_signal': _set_export_installed_signal, 'signal_signature': _set_export_signal_signature, 'scope_rows': _set_export_scope_rows, 'scope_signature': _set_export_scope_signature, '_remember_final': _set_export__remember_final, '_scope_owner': _set_export__scope_owner, '_scope_drop_owner': _set_export__scope_drop_owner, 'scope_signal': _set_export_scope_signal, 'scope_release': _set_export_scope_release, 'scope_retry': _set_export_scope_retry, '_scope_protected': _set_export__scope_protected, 'scope_tick': _set_export_scope_tick, 'evaluate_signal': _set_export_evaluate_signal, 'uses_lut': _set_export_uses_lut, 'baked_only': _set_export_baked_only, 'stop': _set_export_stop, 'collect_retired': _set_export_collect_retired, '_finished': _set_export__finished, 'context': _set_export_context, 'commit_rows': _set_export_commit_rows, '_checked_rows': _set_export__checked_rows, '_write_rows': _set_export__write_rows, '_output_key': _set_export__output_key, '_fallback': _set_export__fallback, '_install_graph': _set_export__install_graph, 'commit_endpoint_rows': _set_export_commit_endpoint_rows, 'refresh_manifest': _set_export_refresh_manifest, '_output_snapshot': _set_export__output_snapshot, '_manifest': _set_export__manifest, 'validate_output': _set_export_validate_output, 'verify_replacement': _set_export_verify_replacement, '_graph_snapshot': _set_export__graph_snapshot, '_restore_graph': _set_export__restore_graph, '_rollback_graph': _set_export__rollback_graph, '_repair_graph': _set_export__repair_graph, '_publication_checkpoint': _set_export__publication_checkpoint, 'installed_matches': _set_export_installed_matches, 'backend_status': _set_export_backend_status, 'draw_error': _set_export_draw_error, 'clear_error': _set_export_clear_error, 'publish': _set_export_publish, '_publish': _set_export__publish, 'redraw': _set_export_redraw, 'cpu_baker': _set_export_cpu_baker, 'acquire_gpu': _set_export_acquire_gpu, 'new_state': _set_export_new_state, 'request': _set_export_request, 'settle': _set_export_settle, 'live_nodes': _set_export_live_nodes, 'tick': _set_export_tick, 'ensure': _set_export_ensure, '_ensure': _set_export__ensure, 'clear': _set_export_clear})
