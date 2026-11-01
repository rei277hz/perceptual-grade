"""Blender-evaluated scope sources sampled before display transforms.

Simple sources use immutable Image samples. Arbitrary native inputs are read
only from an explicitly authorized, retained capture.
"""

from collections.abc import Mapping

from ..integration.components import Component

def create(_application):
    from ...blender_runtime.execution.diagnostics import text as diagnostic_text


    import time
    import logging
    import numpy as np
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader
    from ...perceptual_grade_core.model.scope_analysis import distribution, apply_chain, evaluate_samples, evaluate_linear, evaluate_snapshot
    from ...blender_runtime.execution.diagnostics import log_exception
    nodes = _application.component('graph.nodes')
    from ...perceptual_grade_core.model import linear as cm, tone as tm
    from ...perceptual_grade_core.model import transforms
    from ...perceptual_grade_core.model import color as ch
    from ...perceptual_grade_core.model.scope_color import MEMBERSHIP_SCHEMA
    from ...perceptual_grade_core.scopes.identity import ScopeIdentity, LinearScopeIdentity
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve

    _images={}
    _image_checked={}
    _image_revisions={}
    _source_sizes={}
    _routes={}
    _sample_layouts={}
    _input_status={}
    _source_frames={}
    _sample_sequence=0
    PROBE_INTERVAL=.5
    _sampler=None
    _generation=0
    _errors={}
    EVALUATION_SCHEMA=3


    def _evaluation_identity(stack, tone, params, matrix, scene_space, grading_space,
                             model_identity=None, stage=3):
        """Freeze the numerical route/model used by one full-scope request.

        Regular Grades are evaluated from captured Stage-2 appearance, while a
        Stack still consumes its final baked output.  Keeping the route explicit
        prevents a mutable installed LUT or RNA model from changing a ticket's
        meaning after capture.
        """
        route = 'stack-final-lut' if stack else ('regular-grade-stage2' if stage == 2
                                                 else 'regular-grade-appearance')
        # Forward/LUT schemas and guard policy are part of the evaluator contract;
        # changing either must invalidate a worker cache even when controls match.
        return (route, EVALUATION_SCHEMA, 'forward-schema-14', 'lut-schema-30',
                'hk-boundary-guard-v1', scene_space, grading_space,
                stage, model_identity, tone, params, matrix)


    def _model_identity(model):
        """Codec-safe complete identity for a baked Stack route."""
        entries=tuple((entry.matrix,entry.tone,entry.color,entry.space)
                      for entry in model.entries)
        return ('stack',13,model.space,tuple(zip(entries,model.directions)))


    def clear():
        nonlocal _sampler, _generation
        jobs = _application.component('scopes.jobs')
        clouds = _application.component('scopes.clouds')
        warp_samples = _application.component('ui.warp_samples'); warp_view = _application.component('ui.warp_view')
        jobs.clear()
        _application.component('scopes.automatic').clear()
        _application.component('scopes.capture').clear()
        warp_samples.clear()
        warp_view.clear()
        clouds.clear()
        _application.component('scopes.images').clear()
        _routes.clear();_sample_layouts.clear();_input_status.clear();_source_frames.clear()
        _images.clear();_image_checked.clear();_image_revisions.clear();_source_sizes.clear();_errors.clear();_sampler=None;_generation+=1


    def refresh():
        nonlocal _generation
        jobs = _application.component('scopes.jobs')
        jobs.clear()
        _application.component('scopes.automatic').clear()
        _application.component('scopes.capture').clear_scopes()
        _application.component('scopes.images').invalidate()
        _images.clear();_image_checked.clear();_image_revisions.clear();_source_sizes.clear();_generation+=1


    def channel(node, context=None):
        """Keep an editor's status addressable even when context discovery fails."""
        if isinstance(context,Mapping) and 'consumer' in context:
            consumer=context['consumer']
            return node.as_pointer() if consumer is None else (node.as_pointer(),tuple(consumer))
        # RNA Context exposes keys but does not support mapping operations.
        # Incomplete descriptors retain discovery in the current live editor.
        live_context=bpy.context if context is None or isinstance(context,Mapping) else context
        try:
            editing = _application.component('graph.editing')
            owner=node.stack if editing.is_target(node) else node
            consumer=_application.component('scopes.context').consumer(owner,live_context)
            return node.as_pointer() if consumer is None else (node.as_pointer(),consumer)
        except (ValueError,ReferenceError,RuntimeError):
            editor=tuple(getattr(getattr(live_context,name,None),'as_pointer',lambda:0)()
                         for name in ('window','area','region'))
            return node.as_pointer(),('unresolved',)+editor


    def sampler():
        nonlocal _sampler
        if _sampler is None:
            interface=gpu.types.GPUStageInterfaceInfo('pg_scope_image_uv')
            interface.smooth('VEC2','uv')
            info=gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0,'VEC2','position');info.vertex_in(1,'VEC2','texCoord')
            info.vertex_out(interface);info.sampler(0,'FLOAT_2D','inputImage')
            info.fragment_out(0,'VEC4','fragColor')
            info.vertex_source('void main() { uv = texCoord; gl_Position = vec4(position, 0.0, 1.0); }')
            info.fragment_source('void main() { fragColor = texture(inputImage, uv); }')
            _sampler=gpu.shader.create_from_info(info)
        return _sampler



    def resolve_source(node,space,seen=None,socket_name='Color',dependencies=None,analytical=None,native_context=None):
        route=_application.component('scopes.sources').resolve(node,socket_name,native_context)
        chain,signature=_route_chain(route,space,dependencies,analytical)
        return route.source,chain,signature,route.label,None

    def _route_chain(route,space,dependencies=None,analytical=None):
        chain=[];signature=[]
        spaces=_application.component('graph.spaces')
        for kind,handle in route.steps:
            source=resolve(handle)
            if source is None:raise ReferenceError('An upstream scope Grade was removed')
            if kind=='stack':
                dependency=scope_dependency(source,dependencies)
                signature.append(('stack',dependency['signature']))
                # Defer final-cube acquisition until foreground sampling.
                chain.append(('STACK',handle))
            else:
                model=None if analytical is None else analytical.get(source.as_pointer())
                if model is None:
                    model=transforms.Grade(nodes.settings(source))
                    if analytical is not None:analytical[source.as_pointer()]=model
                before,after=spaces.adapters(space)
                chain.append(('GRADE',model,_array(before),_array(after)))
                signature.append(('grade',(model.settings.matrix,model.settings.tone,model.settings.color,model.settings.space)))
        return chain,tuple(signature)

    def source_route(node,socket_name='Color',context=None):
        route=_application.component('scopes.sources').resolve(node,socket_name,context)
        address=(channel(node,context),socket_name)
        old=_routes.get(address)
        _routes[address]=route
        while len(_routes)>128:_routes.pop(next(iter(_routes)))
        return route

    def _accepted(node,socket_name,context):
        return _application.component('scopes.capture').accepted_descriptor(node,socket_name,context=context)

    def _signature(node,route,socket_name,size,dependencies,analytical,context):
        images=_application.component('scopes.images')
        if route.kind=='native':
            accepted=_accepted(node,socket_name,context)
            version=accepted['version'] if accepted else None
            return ('native',route.structure),version
        companion=_accepted(node,socket_name,context) if _input_status.get(channel(node,context),{}).get('native') else None
        chain,models=_route_chain(route,nodes.working_space(),dependencies,analytical)
        if route.kind=='constant':return ('constant',),('constant',companion.get('_constant_source',route.source) if companion else route.source,models)
        editing=_application.component('graph.editing')
        owner=node.stack if editing.is_target(node) else node
        layout=_sample_layouts.get((channel(owner,context),'Color')) if socket_name!='Color' else None
        key,revision=((companion['_image_key'],companion['version']) if companion and companion.get('_direct_kind')=='image'
                      else images.signature(route.source,layout=layout))
        return ('image',route.source.continuity,route.structure),(key,revision,models)

    def _layout_data(layout):
        return {name:getattr(layout,name) for name in ('mass','x','uv','tiles','ids','key')}

    def _layout_from_data(value):
        from ...perceptual_grade_core.scopes.samples import SampleLayout
        return SampleLayout(**value,tiled=value['key'][0]=='scope-tiles-v1')

    def _reference_compatible(owner,descriptor,context):
        layout=_sample_layouts.get((channel(owner,context),'Color'))
        captured=descriptor.get('_sample_grid')
        return layout is None or (captured is not None and captured['key']==layout.key)

    def scope_input_status(node,context=None):
        address=channel(node,context)
        value=_input_status.get(address)
        if value is None:return dict(native=False,pending=False,state='preparing',error='')
        if not value['native']:return value
        native=_application.component('scopes.capture').status(node,context=context)
        state=_application.component('scopes.automatic').status(node,context)
        return dict(value,**native,native=True) if state is None or native.get('state')=='error' else dict(
            value,**dict(native,state=state,pending=state=='updating'),native=True)

    def native_inputs(node,context=None):
        """Describe source acquisition independently of the selected model."""
        editing=_application.component('graph.editing')
        owner=node.stack if editing.is_target(node) else node
        contexts=_application.component('scopes.context')
        context=contexts.source_context(owner,context)
        reference=_application.component('scopes.reference')
        sockets=['Color']+([reference.SOCKET] if reference.visible(node) else [])
        native=_application.component('scopes.capture')
        inputs=[]
        primary=source_route(owner,context=context)
        routes=[(socket,owner,
                 primary if socket=='Color' else source_route(node,socket,context)) for socket in sockets]
        native_route=any(route.kind=='native' for _,_,route in routes)
        _input_status[channel(node,context)]=dict(native=native_route,pending=False,state='automatic',error='')
        if not native_route:return owner,context,()
        for socket,target,route in routes:
            descriptor=native.describe_source(target,socket,context=context,route=route)
            inputs.append((socket,descriptor,None,socket!='Color'))
        return owner,context,tuple(inputs)

    def source_samples(node,space,socket_name='Color',size=None,force=False,dependencies=None,analytical=None,native_context=None,raw=False,frozen_route=None,asynchronous=False,prepared=None):
        route=frozen_route if frozen_route is not None else source_route(node,socket_name,native_context)
        chain,_=_route_chain(route,nodes.working_space(),dependencies,analytical) if route.kind!='native' and not raw else ([],())
        chain=[('BAKED',*scope_signal(resolve(step[1]),dependencies)) if step[0]=='STACK' else step for step in chain]
        key=(channel(node,native_context),socket_name)
        editing=_application.component('graph.editing')
        owner=node.stack if editing.is_target(node) else node
        layout=_sample_layouts.get((channel(owner,native_context),'Color')) if socket_name!='Color' else None
        companion=_accepted(node,socket_name,native_context) if _input_status.get(channel(node,native_context),{}).get('native') else None
        if prepared is not None:
            rgba,identity,label,layout=prepared
        elif companion is not None and companion.get('_direct_kind')==route.kind:
            rgba,_,label=_application.component('scopes.capture').sample(node,socket_name,descriptor=companion)
            from ...perceptual_grade_core.scopes.samples import ordinary_layout
            grid=companion.get('_sample_grid')
            layout=_layout_from_data(grid) if grid is not None else ordinary_layout(rgba.shape[1],rgba.shape[0])
            identity=(((companion['_image_key'],layout.key),companion['version'],layout.key,_generation)
                      if route.kind=='image' else ('constant',companion['_constant_source'],layout.key,_generation))
        elif route.kind=='native':
            native=_application.component('scopes.capture')
            descriptor=_accepted(node,socket_name,native_context)
            if descriptor is None:raise _application.component('graph.lut').ScopePreparing('Waiting for automatic scope inputs')
            if socket_name!='Color' and not _reference_compatible(owner,descriptor,native_context):
                raise _application.component('graph.lut').ScopePreparing('Preparing the reference sample grid')
            rgba,identity,label=native.sample(node,socket_name,size,descriptor=descriptor,
                exact_size=size is not None and socket_name!= 'Color')
            from ...perceptual_grade_core.scopes.samples import ordinary_layout
            grid=descriptor.get('_sample_grid')
            layout=_layout_from_data(grid) if grid is not None else ordinary_layout(rgba.shape[1],rgba.shape[0])
            chain=[]
        elif route.kind=='constant':
            from ...perceptual_grade_core.scopes.samples import ordinary_layout
            layout=layout or ordinary_layout(*(size or (256,1)))
            rgba=np.broadcast_to(np.asarray(route.source,dtype=float),(*layout.shape,4)).copy()
            identity=('constant',route.source,layout.key,_generation);label=route.label
        else:
            images=_application.component('scopes.images')
            if layout is None and size is not None:
                from ...perceptual_grade_core.scopes.samples import ordinary_layout
                layout=ordinary_layout(*size)
            result=images.read(route.source,layout=layout,force=force,asynchronous=asynchronous)
            rgba,layout=result.rgba,result.layout
            identity=(result.key,result.revision,layout.key,_generation)
            label=result.label
            if result.incomplete:label+=' (incomplete reference)'
        _sample_layouts[key]=layout
        _source_sizes[node.as_pointer(),socket_name]=(rgba.shape[1],rgba.shape[0])
        while len(_source_sizes)>128:_source_sizes.pop(next(iter(_source_sizes)))
        while len(_sample_layouts)>128:_sample_layouts.pop(next(iter(_sample_layouts)))
        return rgba,chain,identity,label


    def _request(node,request,*args,**kwargs):
        """Log unsupported sources once and recover after a successful request."""
        key=node.as_pointer()
        try:
            result=request(node,*args,**kwargs)
        except (ValueError,RuntimeError,ReferenceError,TypeError,OSError,ArithmeticError) as exc:
            message=diagnostic_text(exc)
            if _errors.get(key)!=message:
                log_exception(logging.getLogger(__package__),'Perceptual Grade scopes: %s',message)
                _errors[key]=message
            raise
        _errors.pop(key,None)
        return result


    def get(node,space,view=(),ui=False):
        """Request analysis without sampling pixels, calculating scopes or waiting."""
        jobs = _application.component('scopes.jobs')
        result=_request(node,jobs.request,space,view,ui=ui)
        return select_view(result,view)


    def select_view(result,view=()):
        """Select only an available compact page; never expose legacy point data."""
        if result is not None and '_active_colors' in result:
            if len(view)!=2 or not isinstance(view[0],int):return None
            from types import MappingProxyType
            selected=result['_active_colors'].get(f'{int(view[0])}:{"warp" if view[1] else "jc"}')
            # A pending tone switch must never display another group's raster.
            if selected is None:return None
            base=dict(result);visible=MappingProxyType(base)
            chosen=dict(selected);chosen['_base']=visible
            base['_color']=MappingProxyType(chosen)
            base['_comparison_selected']=chosen.get('_comparison')
            base['_comparison']=base['_comparison_selected']
            return visible
        if len(view)==2 and isinstance(view[0],int):return None
        return result


    def get_tone(node,space,ui=False):
        """Request the regular Grade's Stage-2 result before Stage 3."""
        editing = _application.component('graph.editing')
        return get(node,space,() if editing.is_target(node) else ('tone',),ui=ui)


    def get_linear(node,space,ui=False):
        """Regular Grade's matrix scope; Stack retains its whole-output scope."""
        editing = _application.component('graph.editing')
        if editing.is_target(node):return get(node,space,ui=ui)
        jobs = _application.component('scopes.jobs')
        return _request(node,jobs.request_linear,space,ui=ui)


    def live(node):
        """Resolve RNA membership rather than trusting a removed node's pointer."""
        if node is None:return False
        editing = _application.component('graph.editing')
        try:
            target=editing.is_target(node);owner=node.stack if target else node
            tree=owner.id_data
            member=tree.nodes.get(owner.name)
            found=member is not None and member.as_pointer()==owner.as_pointer()
            return found and (not target or editing.valid(node))
        except (ReferenceError,RuntimeError):return False


    def dirty_samples(node):
        """Visible Image descriptors, including clean reloads and animated inputs."""
        editing=_application.component('graph.editing');reference=_application.component('scopes.reference')
        owner=node.stack if editing.is_target(node) else node
        values=[]
        for target,socket in [(owner,'Color')]+([(node,reference.SOCKET)] if reference.visible(node) else []):
            route=source_route(target,socket)
            if route.kind=='image':values.append(route.source)
        return tuple(values)


    def probe_delay(node):
        images=_application.component('scopes.images')
        delays=[images.probe_delay(value) for value in dirty_samples(node)]
        return min(delays) if delays else None


    def probe_sources(node,space,force=False,capture_context=None):
        """Read source pixels only on the guarded foreground work path."""
        editing=_application.component('graph.editing');reference=_application.component('scopes.reference')
        owner=node.stack if editing.is_target(node) else node
        route=source_route(owner,context=capture_context)
        if route.kind=='native' and _accepted(owner,'Color',capture_context) is None:return
        rgba,_,_,_=source_samples(owner,space,force=force,native_context=capture_context,raw=True)
        if reference.visible(node):
            try:source_samples(node,space,reference.SOCKET,(rgba.shape[1],rgba.shape[0]),
                               force=force,native_context=capture_context,raw=True)
            except (ValueError,ReferenceError,RuntimeError,OSError):pass


    def prepare_sources(spec,force=False,asynchronous=False):
        """Complete one frozen acquisition before binding its numerical revision.

        The active request retains its models and source-frame descriptors while
        later requests replace the pending slot. First-read revision zero is a
        discovery placeholder, never the identity of a published sample.
        """
        node=resolve(spec['node'])
        if node is None:raise ReferenceError('Retired scope acquisition')
        owner=node.stack if spec.get('stack') else node
        reference=_application.component('scopes.reference')
        primary=None
        acquired=spec.setdefault('_scope_prepared_sources',{})
        for index,(route,socket) in enumerate(zip(spec['_source_routes'],('Color',reference.SOCKET))):
            if route is None or (index and not spec['comparison']):continue
            size=None if not index else (primary.shape[1],primary.shape[0])
            rgba,_,identity,label=source_samples(owner if not index else node,spec['space'],socket,size,
                force=force,dependencies=spec['_dependencies'],analytical=dict(spec.get('_grades',())),
                native_context=spec.get('_capture_context'),raw=True,frozen_route=route,
                asynchronous=asynchronous,prepared=acquired.get(index))
            if index not in acquired:
                acquired[index]=(_array(rgba),identity,label,
                    _sample_layouts[channel(owner if not index else node,spec.get('_capture_context')),socket])
            if not index:primary=rgba
        return bind_sources(spec,complete=True)


    def bind_sources(spec,complete=False):
        """Project only verified acquisitions onto a frozen request's identity.

        Partial primary/reference acquisition may advance discovery revision
        zero. Its own accepted receipt must not look like a source replacement.
        """
        acquired=spec.get('_scope_prepared_sources',{})
        if not acquired and not complete:return spec
        from dataclasses import replace
        linear=spec.get('scope_stage')=='LINEAR'
        versions=[spec['source_key'][0] if linear else spec['key'].version,
                  spec['reference_key'][0] if linear and spec.get('reference_key') is not None
                  else None if linear else spec['key'].reference_version]
        for index,route in enumerate(spec['_source_routes']):
            if route is not None and route.kind=='image' and index in acquired:
                identity=acquired[index][1]
                versions[index]=(identity[0][0],identity[1],versions[index][-1])
        def pixels(route,version):
            if route is None or version is None:return None
            if route.kind=='image' and route.source.animated:return None
            return version if route.kind=='native' else version[:-1]
        raw=tuple(pixels(route,version) for route,version in zip(spec['_source_routes'],versions))
        result=dict(spec,context=(spec['context'][0],raw))
        if complete:result['_scope_unread']=False
        if linear:
            source=(versions[0],spec['scene_space'],spec['space'],_generation)
            ref=(versions[1],source) if spec['comparison'] else None
            result.update(source_key=source,reference_key=ref,
                key=replace(spec['key'],source=source,reference=ref),
                scope_source=(spec['scope_source'][0],raw),
                _scope_display_source=result['context'])
        else:
            result.update(key=replace(spec['key'],version=versions[0],reference_version=versions[1]),
                scope_source=(spec['scope_source'][0],raw,*spec['scope_source'][2:]))
            display=spec['_scope_display_source']
            result['_scope_display_source']=((*display[0][:-1],raw),*display[1:])
            result['_scope_compatibility']=(result['context'],result['scope_source'],spec['view'],
                spec.get('_scope_demands',()),tuple(v['key'] for v in spec.get('_warp_views',())),
                spec.get('_scope_backend'))
            if '_warp_analysis_key' in spec:result['_warp_analysis_key']=result['key'].analysis()
        return result


    def _scope_structure(node):
        """Grade bindings and coordinate spaces are structural, controls are not."""
        stacks = _application.component('graph.stacks')
        if stacks.is_stack(node):
            return tuple((item.key,item.grade.as_pointer() if item.grade else None,
                          item.enabled,item.direction,nodes.grading_space(item.grade) if item.grade else None)
                         for item in node.pg_stack_entries)
        # First numeric edits can materialize a shared Grade datablock. That is
        # still the same displayed signal and must not invalidate its components.
        return nodes.grading_space(node)


    def native_dependencies(node,socket_name,dependencies,native_context=None):
        """Pin consumed live PG outputs with an iterative, occurrence-aware walk."""
        stacks=_application.component('graph.stacks')
        seen=set()
        parents=[]
        if native_context is not None and native_context.get('group_path'):
            kind=native_context['owner_kind']
            if kind=='compositor':
                root=bpy.data.scenes[native_context['scene']].compositing_node_group
            else:
                root=(bpy.data.materials if kind=='material' else bpy.data.worlds)[native_context['owner_name']].node_tree
            for name in native_context['group_path']:
                instance=root.nodes[name];parents.append(instance);root=instance.node_tree
        def matching(sockets,identifier,name):
            return next((item for item in sockets if item.identifier==identifier),sockets.get(name))
        incoming={}
        def input_links(socket):
            tree=socket.id_data
            pointer=tree.as_pointer()
            if pointer not in incoming:
                links={}
                for link in tree.links:
                    links.setdefault(link.to_socket.as_pointer(),[]).append(link)
                incoming[pointer]=links
            return incoming[pointer].get(socket.as_pointer(),())
        pending=[('input',node.inputs.get(socket_name),tuple(parents))]
        try:
            while pending:
                kind,socket,occurrence=pending.pop()
                if socket is None:continue
                path=tuple(parent.as_pointer() for parent in occurrence)
                if kind=='input':
                    key=(socket.as_pointer(),path)
                    if key in seen:continue
                    seen.add(key)
                    # Reverse pushes preserve the original depth-first order.
                    pending.extend(('output',link.from_socket,occurrence)
                                   for link in reversed(input_links(socket)) if not link.is_muted)
                    continue
                source=socket.node
                key=(source.as_pointer(),socket.identifier,path)
                if key in seen:continue
                seen.add(key)
                if source.mute:
                    bypass=tuple(link.from_socket for link in source.internal_links if link.to_socket==socket)
                    if bypass:
                        pending.extend(('input',value,occurrence) for value in reversed(bypass))
                        continue
                if ((nodes.is_grade(source) or stacks.is_stack(source))
                        and not _application.component('graph.lut').is_adopted(source)):
                    # These outputs do not evaluate the live grading model.
                    # Capture fingerprinting tracks the bound baked Image.
                    if socket.name=='Baked Color':continue
                    if socket.name=='Alpha':
                        pending.append(('input',source.inputs.get('Alpha'),occurrence))
                        continue
                    scope_dependency(source,dependencies)['native_source']=True
                    pending.append(('input',source.inputs.get('Color'),occurrence))
                    continue
                if source.bl_idname=='NodeReroute':
                    pending.append(('input',source.inputs[0],occurrence));continue
                if source.bl_idname=='NodeGroupInput' and occurrence:
                    external=matching(occurrence[-1].inputs,socket.identifier,socket.name)
                    pending.append(('input',external,occurrence[:-1]));continue
                child=getattr(source,'node_tree',None)
                if child is not None:
                    outputs=[item for item in child.nodes if item.bl_idname=='NodeGroupOutput']
                    active=next((item for item in outputs if getattr(item,'is_active_output',False)),
                                outputs[0] if outputs else None)
                    if active is None:raise ValueError('Scope source group has no active output')
                    pending.append(('input',matching(active.inputs,socket.identifier,socket.name),
                                    occurrence+(source,)))
                    continue
                pending.extend(('input',value,occurrence) for value in reversed(tuple(source.inputs))
                               if value.is_linked)
        finally:
            pending.clear()
            incoming.clear()
            parents.clear()


    def source_signature(node,socket_name='Color',seen=None,size=None,dependencies=None,analytical=None,native_context=None):
        route=source_route(node,socket_name,native_context)
        return _signature(node,route,socket_name,size,dependencies,analytical,native_context)


    def signal_signature(node):
        lut = _application.component('graph.lut')
        return lut.signal_signature(node)


    def scope_dependency(node,dependencies=None):
        """Freeze a numerical core and its endpoint rows, independently of previews."""
        lut = _application.component('graph.lut'); spaces = _application.component('graph.spaces')
        f32 = _application.component('graph.optimize').f32
        key=node.as_pointer()
        if dependencies is not None and key in dependencies:return dependencies[key]
        model=nodes.bake_model(node)
        before,after=spaces.adapters(nodes.working_space())
        if nodes.is_grade(node):
            first=lut.cm.TO_LUT@nodes.linear_matrix(node)@before
            last=after@lut.cm.FROM_LUT
        else:
            first=model.input_matrix@lut.cm.TO_LUT@before
            last=after@lut.cm.FROM_LUT@model.output_matrix
        rows=tuple(_array([[f32(value) for value in row] for row in matrix]) for matrix in (first,last))
        value=dict(node=retain(node,'scope.dependency'),model=model,rows=rows,
            signature=lut.scope_signature(node,model,rows),cube_signature=lut.cm.signature(model))
        if dependencies is not None:dependencies[key]=value
        return value


    def scope_signal(node,dependencies=None):
        lut = _application.component('graph.lut')
        dependency=scope_dependency(node,dependencies)
        if 'signal' not in dependency:
            dependency['signal']=lut.scope_signal(
                node,dependency['model'],dependency['rows'],owner=id(dependency))
        return dependency['signal']


    def provenance(dependencies):
        return tuple(sorted({(item['cube_signature'],65) for item in dependencies.values()}))


    def captured_provenance(chains=(),signal=None):
        values=[]
        if signal is not None and signal[0] is not None:values.append(signal[0])
        for chain in chains:
            values.extend(step[1] for step in chain if step[0]=='BAKED')
        return tuple(sorted({(cube.signature,cube.size) for cube in values}))


    def source_revision(node,socket_name='Color',seen=None):
        route=source_route(node,socket_name)
        if route.kind=='native':
            accepted=_accepted(node,socket_name,None)
            return accepted['version'] if accepted else None
        if route.kind=='constant':return route.source
        editing=_application.component('graph.editing')
        owner=node.stack if editing.is_target(node) else node
        layout=_sample_layouts.get((channel(owner),'Color')) if socket_name!='Color' else None
        return _application.component('scopes.images').signature(route.source,layout=layout)


    def release(spec):
        """Release the protected final-cube demands owned by one scope snapshot."""
        lut = _application.component('graph.lut')
        for item in spec.get('_dependencies',{}).values():
            node=resolve(item['node'])
            if node is not None:lut.scope_release(node,item['model'],owner=id(item))
        spec.pop('_scope_prepared_sources',None)


    def acquisition_retired(spec):
        """A cancelled channel retains its lease until shared decodes retire."""
        images=_application.component('scopes.images')
        return all(images.retired(route.source) for index,route in enumerate(spec.get('_source_routes',()))
                   if route is not None and route.kind=='image'
                   and index not in spec.get('_scope_prepared_sources',{}))


    def _input_description(node,owner,context,dependencies,analytical):
        reference=_application.component('scopes.reference')
        context=_application.component('scopes.context').source_context(owner,context)
        primary=source_route(owner,context=context)
        wanted=reference.visible(node)
        ref=source_route(node,reference.SOCKET,context) if wanted else None
        native=primary.kind=='native' or bool(ref and ref.kind=='native')
        _input_status[channel(node,context)]=dict(native=native,pending=False,state='automatic',error='')
        accepted=_accepted(owner,'Color',context) if native else None
        if native and accepted is None:
            raise _application.component('graph.lut').ScopePreparing('Waiting for automatic scope inputs')
        ref_accepted=_accepted(node,reference.SOCKET,context) if ref and native else None
        available=wanted and (not native or ref_accepted is not None)
        layout=(_layout_from_data(accepted['_sample_grid']) if accepted and accepted.get('_sample_grid') is not None
                else _sample_layouts.get((channel(owner,context),'Color')))
        if layout is not None:_sample_layouts[channel(owner,context),'Color']=layout
        size=None if layout is None else (layout.shape[1],layout.shape[0])
        if available and native:
            available=_reference_compatible(owner,ref_accepted,context)
            if not available:
                _application.component('scopes.capture').mark_stale(owner,context,
                    reason='Primary sampling grid changed')
        if available and native:
            if size is not None:
                try:_application.component('scopes.capture').sample(node,reference.SOCKET,size,
                    descriptor=ref_accepted,exact_size=True)
                except ValueError:available=False

        pc,pv=_signature(owner,primary,'Color',None,dependencies,analytical,context)
        rc=rv=None
        if available:
            rc,rv=_signature(node,ref,reference.SOCKET,size,
                             dependencies,analytical,context)
        native_provenance=tuple(sorted(set(item for descriptor in (accepted,ref_accepted) if descriptor
            for item in descriptor.get('_scope_provenance',()))))
        frames=tuple((descriptor['_image_source']['frame'] if descriptor and descriptor.get('_direct_kind')=='image' else route.source.frame)
                     for route,descriptor in ((primary,accepted),(ref,ref_accepted))
                     if route and route.kind=='image' and route.source.animated)
        unread=(primary.kind=='image' and pv[1]==0) or bool(available and ref.kind=='image' and rv[1]==0)
        # Upstream analytical controls change the request, while its pixel domain
        # stays compatible during a gesture. Native versions already freeze the
        # complete evaluated input and have no separate analytical suffix.
        def compatible_pixels(route,version):
            if route is None or version is None:return None
            if route.kind=='image' and route.source.animated:return None
            return version if route.kind=='native' else version[:-1]
        pixels=(compatible_pixels(primary,pv),compatible_pixels(ref,rv))
        return dict(context=context,primary=primary,reference=ref,source_context=pc,version=pv,
                    ref_context=rc,ref_version=rv,comparison=available,native_provenance=native_provenance,
                    animated=bool(frames),frames=frames,unread=unread,pixels=pixels,
                    scene_frame=next((item['view']['frame'] for item in (accepted,ref_accepted) if item is not None),_scene_frame(context)))

    def describe(node,space,view=(),capture_context=None):
        editing=_application.component('graph.editing')
        if not live(node):raise ReferenceError('The scope node was removed')
        stack=editing.is_target(node);owner=node.stack if stack else node
        scene_space=nodes.working_space();space=nodes.grading_space(node)
        dependencies={};analytical={}
        inputs=_input_description(node,owner,capture_context,dependencies,analytical)
        own=scope_dependency(owner,dependencies) if stack else None
        context=(inputs['source_context'],_scope_structure(owner))
        comparison=inputs['comparison'];ref_context=inputs['ref_context']
        version=inputs['version'];ref_version=inputs['ref_version']
        tone=nodes.read_tone(node);matrix=nodes.grades.matrix(node)
        stage=2 if (not stack and view==('tone',)) else 3
        params=None if stage==2 else nodes.read_chromatic(node)
        request_params=ch.Parameters() if stage==2 else params
        evaluation=_evaluation_identity(stack,tone,request_params,matrix,scene_space,space,
                                        _model_identity(own['model']) if own is not None else None,stage)
        structural=(context,ref_context,scene_space,space,stage,_generation)
        raw=inputs['pixels']
        identity=(structural,raw)
        display_source=(context,ref_context,scene_space,space,_generation,raw)
        key=ScopeIdentity(version=version,reference_version=ref_version,comparison=comparison,
                          matrix=matrix,tone=tone,request_params=request_params,
                          stack_signature=own['signature'] if own is not None else None,
                          generation=_generation,stage=stage)
        spec=dict(_capture_context=inputs['context'],_scope_channel=channel(node,inputs['context']),
                  node=retain(node,'scope.full'),space=space,scene_space=scene_space,view=view,stack=stack,
                  context=identity,key=key,tone=tone,matrix=matrix,params=params,comparison=comparison,
                  _scope_display_source=(display_source,'scope-coordinates-2',MEMBERSHIP_SCHEMA),
                  _dependencies=dependencies,scope_provenance=tuple(sorted(set(provenance(dependencies))|set(inputs['native_provenance']))),
                  scope_resolution=65,_scope_evaluation=evaluation,stage=stage,_grades=tuple(analytical.items()),
                  _scope_history=True,_scope_frames=inputs['frames'],
                  _scope_scene_frame=inputs['scene_frame'],
                  _scope_unread=inputs['unread'],
                  _source_routes=(inputs['primary'],inputs['reference']),
                  scope_source=(structural,raw,comparison,stage),
                  linear=nodes.read_parameters(node),linear_matrix=_array(nodes.linear_matrix(node)))
        spec=_application.component('scopes.clouds').decorate(spec,node)
        spec['_scope_compatibility']=(identity,spec['scope_source'],view,
            spec.get('_scope_demands',()),tuple(v['key'] for v in spec.get('_warp_views',())),
            spec.get('_scope_backend'))
        return spec

    def describe_linear(node,space,capture_context=None):
        if not nodes.is_grade(node) or not live(node):raise ReferenceError('The scope Grade was removed')
        scene_space=nodes.working_space();space=nodes.grading_space(node)
        dependencies={};analytical={}
        inputs=_input_description(node,node,capture_context,dependencies,analytical)
        context=(inputs['source_context'],_scope_structure(node))
        comparison=inputs['comparison'];ref_context=inputs['ref_context']
        version=inputs['version'];ref_version=inputs['ref_version']
        # Computation keys do not contain editor, target-node or group pointers.
        source_key=(version,scene_space,space,_generation)
        reference_key=(ref_version,source_key) if comparison else None
        matrix=nodes.grades.matrix(node)
        evaluation=_evaluation_identity(False,None,None,matrix,scene_space,space,stage=1)
        structural=(context,ref_context,comparison,scene_space,space,_generation)
        raw=inputs['pixels']
        identity=(structural,raw)
        return dict(_capture_context=inputs['context'],_scope_channel=channel(node,inputs['context']),
            node=retain(node,'scope.linear'),space=space,scene_space=scene_space,source_key=source_key,
            reference_key=reference_key,comparison=comparison,context=identity,
            _scope_display_source=identity,key=LinearScopeIdentity(source=source_key,reference=reference_key,matrix=matrix),matrix=matrix,
            _dependencies=dependencies,scope_provenance=tuple(sorted(set(provenance(dependencies))|set(inputs['native_provenance']))),
            scope_resolution=65,_scope_evaluation=evaluation,_grades=tuple(analytical.items()),scope_stage='LINEAR',
            _scope_history=True,_scope_frames=inputs['frames'],
            _scope_scene_frame=inputs['scene_frame'],
            _scope_unread=inputs['unread'],
            _source_routes=(inputs['primary'],inputs['reference']),
            scope_source=(structural,raw),
            linear_matrix=_array(nodes.linear_matrix(node)))

    def _scene_frame(context):
        scene=bpy.data.scenes.get(context.get('scene','')) if context else None
        return int((scene or bpy.context.scene).frame_current)


    def _array(value):
        result=np.array(value,copy=True);result.setflags(write=False);return result


    def _signal(signal):
        from ...perceptual_grade_core.model import lut
        cube,before,after=signal
        if cube is not None:cube=lut.Cube(_array(cube.values),cube.signature,cube.space,cube.size)
        return cube,_array(before),_array(after)


    def _chain(chain):
        return tuple(('BAKED',*_signal(step[1:])) if step[0]=='BAKED' else step for step in chain)


    def _native_ready(spec):
        """Finish requested producers; adopted outputs are native scene inputs."""
        if not spec.get('_native_admission'):return
        lut=_application.component('graph.lut')
        selected=resolve(spec['node'])
        selected=selected.stack if spec.get('stack') else selected
        for dependency in tuple(spec['_dependencies'].values()):
            source=resolve(dependency['node'])
            if source is None:raise ReferenceError('Upstream scope Grade was removed')
            if lut.is_adopted(source):continue
            if scope_dependency(source)['signature']!=dependency['signature']:
                raise ValueError('Upstream Grade changed before capture; refresh again')
            if source==selected and spec.get('stack') and not dependency.get('native_source'):continue
            scope_signal(source,spec['_dependencies'])
            try:lut.validate_output(source,dependency['model'],final=True)
            except (ValueError,RuntimeError):
                lut.request(source,dependency['model'])
                raise lut.ScopePreparing('Preparing exact upstream Grade output')


    def _validate_native_source(spec,identity,reference=False):
        route=spec['_source_routes'][1 if reference else 0]
        if route is None or route.kind=='constant':return
        if 'source_key' in spec:
            expected=spec['reference_key'][0] if reference else spec['source_key'][0]
        else:expected=spec['key'].reference_version if reference else spec['key'].version
        if route.kind=='image':
            if identity[0][0]!=expected[0] or identity[1]!=expected[1]:
                raise ValueError('The sampled Image changed during scope acquisition')
            return
        if len(identity)<4 or identity[0]!='native-scope-v1' or identity[2]!=expected:
            raise _application.component('graph.lut').ScopePreparing('The captured Blender source changed')

    def capture_linear(spec,cache):
        """Capture source revisions once; never snapshot this Grade's own LUT."""
        spaces = _application.component('graph.spaces')
        reference = _application.component('scopes.reference')
        node=resolve(spec['node']); analytical=dict(spec.get('_grades',()))
        if node is None:raise ReferenceError('Retired scope capture')
        def source(key,socket_name='Color',size=None):
            if key in cache:return cache[key]
            rgba,chain,identity,label=source_samples(node,spec['space'],socket_name,size,
                dependencies=spec['_dependencies'],analytical=analytical,native_context=spec.get('_capture_context'),
                frozen_route=spec['_source_routes'][0 if socket_name=='Color' else 1],
                asynchronous=spec.get('_scope_asynchronous',False),
                prepared=spec.get('_scope_prepared_sources',{}).get(0 if socket_name=='Color' else 1))
            _validate_native_source(spec,identity,reference=socket_name!="Color")
            return dict(key=key,rgba=_array(rgba),chain=_chain(chain),label=label,
                sample_grid=_layout_data(_sample_layouts[channel(node,spec.get('_capture_context')),socket_name]),
                to_grade=_array(spaces.adapters(spec['scene_space'])[0]),
                _scope_provenance=tuple(sorted(set(spec['scope_provenance'])|
                    set(captured_provenance((chain,))))))
        primary=source(spec['source_key']);comparison=None
        if spec['comparison']:
            pixels=primary['rgb'] if 'rgb' in primary else primary['rgba']
            height,width=pixels.shape[:2]
            comparison=source(spec['reference_key'],reference.SOCKET,(width,height))
        # A graph/model edit can supersede this frozen request while its final LUT
        # dependencies are being prepared. Keep the captured request's identity;
        # the coordinator decides whether a lagging drag result may be displayed.
        return dict(key=spec['key'],source=primary,comparison=comparison,space=spec['space'],
            scene_space=spec['scene_space'],matrix=spec['linear_matrix'],
            _scope_evaluation=spec.get('_scope_evaluation',()),
            _scope_provenance=tuple(sorted(set(primary['_scope_provenance'])|
                set(comparison['_scope_provenance'] if comparison else ()))))




    def cache_id(spec):
        """Raw frozen source identity, independent of this Grade's controls."""
        import hashlib
        key=spec['key']
        value=('scope-source-v2',spec['scene_space'],spec['space'],key.source_projection(spec['stack']),
               spec['scope_provenance'])
        return hashlib.sha256(repr(value).encode('utf8')).hexdigest()


    def capture(spec,seed=None):
        """Main-thread-only snapshot: no Blender objects are passed to the worker."""
        spaces = _application.component('graph.spaces'); lut = _application.component('graph.lut')
        reference = _application.component('scopes.reference')
        node=resolve(spec['node'])
        if node is None:raise ReferenceError('Retired scope capture')
        owner=node.stack if spec['stack'] else node
        dependencies=spec['_dependencies']; analytical=dict(spec.get('_grades',()))
        if seed is None:
            rgba,chain,identity,label=source_samples(owner,spec['space'],dependencies=dependencies,analytical=analytical,native_context=spec.get('_capture_context'),frozen_route=spec['_source_routes'][0],asynchronous=spec.get('_scope_asynchronous',False),prepared=spec.get('_scope_prepared_sources',{}).get(0))
            _validate_native_source(spec,identity)
            signal=_signal(scope_signal(owner,dependencies)) if spec['stack'] else None
            comparison=None
            if spec['comparison']:
                h,w=rgba.shape[:2]
                ref,ref_chain,ref_identity,_=source_samples(node,spec['space'],reference.SOCKET,(w,h),
                    dependencies=dependencies,analytical=analytical,native_context=spec.get('_capture_context'),frozen_route=spec['_source_routes'][1],asynchronous=spec.get('_scope_asynchronous',False),prepared=spec.get('_scope_prepared_sources',{}).get(1))
                _validate_native_source(spec,ref_identity,reference=True)
                comparison=(_array(ref),_chain(ref_chain))
            actual=captured_provenance((chain,ref_chain) if comparison is not None else (chain,),signal)
            source=dict(rgba=_array(rgba),chain=_chain(chain),label=label,signal=signal,
                sample_grid=_layout_data(_sample_layouts[channel(owner,spec.get('_capture_context')),'Color']),
                comparison=comparison,to_grade=_array(spaces.adapters(spec['scene_space'])[0]),
                _scope_provenance=tuple(sorted(set(spec['scope_provenance'])|set(actual))))
        else:
            source={name:seed[name] for name in ('rgba','chain','label','signal','comparison','to_grade','sample_grid','_scope_provenance')}
        key=spec['key']
        base_key=key.capture_projection(None,None,spec['scene_space'],spec['space'])
        snapshot = dict(source,base_key=base_key,key=spec['key'],stack=spec['stack'],space=spec['scene_space'],
                    grading_space=spec['space'],params=spec['params'],view=spec['view'],tone=spec['tone'],
                    stage=spec.get('stage',3),
                    linear=spec['linear'],matrix=spec['linear_matrix'],
                    _scope_evaluation=spec.get('_scope_evaluation',()))
        if '_scope_demands' in spec:
            snapshot['_scope_demands']=spec['_scope_demands']
            snapshot['cache_id']=cache_id(spec)
        if '_warp_views' in spec:
            snapshot['_warp_views'] = spec['_warp_views']
            snapshot['_warp_analysis_key'] = spec['_warp_analysis_key']
        return snapshot



    def _set_scope_input_status(value):
        nonlocal scope_input_status
        scope_input_status=value

    def _set_export_EVALUATION_SCHEMA(value):
        nonlocal EVALUATION_SCHEMA
        EVALUATION_SCHEMA = value

    def _set_export_PROBE_INTERVAL(value):
        nonlocal PROBE_INTERVAL
        PROBE_INTERVAL = value

    def _set_export__errors(value):
        nonlocal _errors
        _errors = value

    def _set_export__generation(value):
        nonlocal _generation
        _generation = value

    def _set_export__image_checked(value):
        nonlocal _image_checked
        _image_checked = value

    def _set_export__image_revisions(value):
        nonlocal _image_revisions
        _image_revisions = value

    def _set_export__images(value):
        nonlocal _images
        _images = value

    def _set_export__sample_sequence(value):
        nonlocal _sample_sequence
        _sample_sequence = value

    def _set_export__sampler(value):
        nonlocal _sampler
        _sampler = value

    def _set_export__source_sizes(value):
        nonlocal _source_sizes
        _source_sizes = value

    def _set_export__evaluation_identity(value):
        nonlocal _evaluation_identity
        _evaluation_identity = value

    def _set_export__model_identity(value):
        nonlocal _model_identity
        _model_identity = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    def _set_export_refresh(value):
        nonlocal refresh
        refresh = value

    def _set_export_sampler(value):
        nonlocal sampler
        sampler = value

    def _set_export_resolve_source(value):
        nonlocal resolve_source
        resolve_source = value

    def _set_export_source_samples(value):
        nonlocal source_samples
        source_samples = value

    def _set_export__request(value):
        nonlocal _request
        _request = value

    def _set_export_get(value):
        nonlocal get
        get = value

    def _set_export_select_view(value):
        nonlocal select_view
        select_view = value

    def _set_export_get_tone(value):
        nonlocal get_tone
        get_tone = value

    def _set_export_get_linear(value):
        nonlocal get_linear
        get_linear = value

    def _set_export_live(value):
        nonlocal live
        live = value

    def _set_export_dirty_samples(value):
        nonlocal dirty_samples
        dirty_samples = value

    def _set_export_probe_delay(value):
        nonlocal probe_delay
        probe_delay = value

    def _set_export_probe_sources(value):
        nonlocal probe_sources
        probe_sources = value

    def _set_export__scope_structure(value):
        nonlocal _scope_structure
        _scope_structure = value

    def _set_export_native_dependencies(value):
        nonlocal native_dependencies
        native_dependencies = value

    def _set_export_source_signature(value):
        nonlocal source_signature
        source_signature = value

    def _set_export_signal_signature(value):
        nonlocal signal_signature
        signal_signature = value

    def _set_export_scope_dependency(value):
        nonlocal scope_dependency
        scope_dependency = value

    def _set_export_scope_signal(value):
        nonlocal scope_signal
        scope_signal = value

    def _set_export_provenance(value):
        nonlocal provenance
        provenance = value

    def _set_export_captured_provenance(value):
        nonlocal captured_provenance
        captured_provenance = value

    def _set_export_source_revision(value):
        nonlocal source_revision
        source_revision = value

    def _set_export_release(value):
        nonlocal release
        release = value

    def _set_export_describe(value):
        nonlocal describe
        describe = value

    def _set_export_describe_linear(value):
        nonlocal describe_linear
        describe_linear = value

    def _set_export__array(value):
        nonlocal _array
        _array = value

    def _set_export__signal(value):
        nonlocal _signal
        _signal = value

    def _set_export__chain(value):
        nonlocal _chain
        _chain = value

    def _set_export_capture_linear(value):
        nonlocal capture_linear
        capture_linear = value

    def _set_export_cache_id(value):
        nonlocal cache_id
        cache_id = value

    def _set_export_capture(value):
        nonlocal capture
        capture = value

    def _set_export_time(value):
        nonlocal time
        time = value

    def _set_export_bpy(value):
        nonlocal bpy
        bpy = value

    return Component(readers={
        "__name__": lambda: __name__,
        'EVALUATION_SCHEMA': lambda: EVALUATION_SCHEMA,
        'LinearScopeIdentity': lambda: LinearScopeIdentity,
        'MEMBERSHIP_SCHEMA': lambda: MEMBERSHIP_SCHEMA,
        'PROBE_INTERVAL': lambda: PROBE_INTERVAL,
        'ScopeIdentity': lambda: ScopeIdentity,
        '_array': lambda: _array,
        '_chain': lambda: _chain,
        '_errors': lambda: _errors,
        '_evaluation_identity': lambda: _evaluation_identity,
        '_generation': lambda: _generation,
        '_image_checked': lambda: _image_checked,
        '_image_revisions': lambda: _image_revisions,
        '_images': lambda: _images,
        '_model_identity': lambda: _model_identity,
        '_request': lambda: _request,
        '_sample_sequence': lambda: _sample_sequence,
        '_sampler': lambda: _sampler,
        '_scope_structure': lambda: _scope_structure,
        '_signal': lambda: _signal,
        '_source_sizes': lambda: _source_sizes,
        'apply_chain': lambda: apply_chain,
        'batch_for_shader': lambda: batch_for_shader,
        'bpy': lambda: bpy,
        'cache_id': lambda: cache_id,
        'capture': lambda: capture,
        'channel': lambda: channel,
        'capture_linear': lambda: capture_linear,
        'captured_provenance': lambda: captured_provenance,
        'ch': lambda: ch,
        'clear': lambda: clear,
        'cm': lambda: cm,
        'describe': lambda: describe,
        'describe_linear': lambda: describe_linear,
        'diagnostic_text': lambda: diagnostic_text,
        'dirty_samples': lambda: dirty_samples,
        'distribution': lambda: distribution,
        'evaluate_linear': lambda: evaluate_linear,
        'evaluate_samples': lambda: evaluate_samples,
        'evaluate_snapshot': lambda: evaluate_snapshot,
        'get': lambda: get,
        'get_linear': lambda: get_linear,
        'get_tone': lambda: get_tone,
        'gpu': lambda: gpu,
        'live': lambda: live,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'nodes': lambda: nodes,
        'np': lambda: np,
        'probe_delay': lambda: probe_delay,
        'probe_sources': lambda: probe_sources,
        'prepare_sources': lambda: prepare_sources,
        'bind_sources': lambda: bind_sources,
        'acquisition_retired': lambda: acquisition_retired,
        'provenance': lambda: provenance,
        'refresh': lambda: refresh,
        'release': lambda: release,
        'resolve': lambda: resolve,
        'resolve_source': lambda: resolve_source,
        'retain': lambda: retain,
        'sampler': lambda: sampler,
        'scope_dependency': lambda: scope_dependency,
        'scope_signal': lambda: scope_signal,
        'select_view': lambda: select_view,
        'signal_signature': lambda: signal_signature,
        'source_revision': lambda: source_revision,
        'source_samples': lambda: source_samples,
        'native_dependencies': lambda: native_dependencies,
        'source_signature': lambda: source_signature,
        'source_route': lambda: source_route,
        'native_inputs': lambda: native_inputs,
        'scope_input_status': lambda: scope_input_status,
        'time': lambda: time,
        'tm': lambda: tm,
        'transforms': lambda: transforms,
    }, writers={
        'scope_input_status':_set_scope_input_status,
        'EVALUATION_SCHEMA': _set_export_EVALUATION_SCHEMA,
        'PROBE_INTERVAL': _set_export_PROBE_INTERVAL,
        'time': _set_export_time,
        'bpy': _set_export_bpy,
        '_errors': _set_export__errors,
        '_generation': _set_export__generation,
        '_image_checked': _set_export__image_checked,
        '_image_revisions': _set_export__image_revisions,
        '_images': _set_export__images,
        '_sample_sequence': _set_export__sample_sequence,
        '_sampler': _set_export__sampler,
        '_source_sizes': _set_export__source_sizes,
    '_evaluation_identity': _set_export__evaluation_identity, '_model_identity': _set_export__model_identity, 'clear': _set_export_clear, 'refresh': _set_export_refresh, 'sampler': _set_export_sampler, 'resolve_source': _set_export_resolve_source, 'source_samples': _set_export_source_samples, '_request': _set_export__request, 'get': _set_export_get, 'select_view': _set_export_select_view, 'get_tone': _set_export_get_tone, 'get_linear': _set_export_get_linear, 'live': _set_export_live, 'dirty_samples': _set_export_dirty_samples, 'probe_delay': _set_export_probe_delay, 'probe_sources': _set_export_probe_sources, '_scope_structure': _set_export__scope_structure, 'native_dependencies': _set_export_native_dependencies, 'source_signature': _set_export_source_signature, 'signal_signature': _set_export_signal_signature, 'scope_dependency': _set_export_scope_dependency, 'scope_signal': _set_export_scope_signal, 'provenance': _set_export_provenance, 'captured_provenance': _set_export_captured_provenance, 'source_revision': _set_export_source_revision, 'release': _set_export_release, 'describe': _set_export_describe, 'describe_linear': _set_export_describe_linear, '_array': _set_export__array, '_signal': _set_export__signal, '_chain': _set_export__chain, 'capture_linear': _set_export_capture_linear, 'cache_id': _set_export_cache_id, 'capture': _set_export_capture})
