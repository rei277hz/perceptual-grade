"""Foreground coordination for helper-owned scope textures and statistics."""

from ..integration.components import Component

def create(_application):
    from collections import OrderedDict
    from dataclasses import replace
    from types import MappingProxyType
    from itertools import count
    from ..runtime.protocol import Protocol
    import logging
    import time
    from ...perceptual_grade_core.jobs.codec import readonly, freeze_scope_request, thaw_scope_result
    from ...blender_runtime.execution.diagnostics import log_exception; from ...blender_runtime.execution.diagnostics import text
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve

    LOG=logging.getLogger(__package__)
    _states=OrderedDict()
    _running=None
    _linear_states=OrderedDict()
    _linear_running=None
    _linear_pending=None
    _linear_sources=OrderedDict()
    _consumers={}
    _request_errors=OrderedDict()
    _scope_sources=OrderedDict()
    _scope_receipts={}
    _consumer_scan_due=0.
    _service={'full':[], 'linear':[]}
    _service_start={'full':0, 'linear':0}
    _retiring_cycles=[]
    _retiring_captures=[]
    _protocol_requests=count(1)


    def _capture_protocol(spec):
        """Own one analytical request independently of reusable native pixels."""
        protocol=spec.get('_scope_protocol')
        if protocol is None:
            scheduler=_application.component('graph.scheduler')
            sampling=_application.component('scopes.sampling')
            native_source=((spec['source_key'],spec.get('reference_key'),spec['scope_provenance'])
                if spec.get('scope_stage')=='LINEAR' else sampling.cache_id(spec))
            source=scheduler.identity(('scope-native-source',native_source))
            protocol=Protocol(_application,'scope',identity=dict(
                request=next(_protocol_requests),source=source,generation=1))
            protocol.require('request','capture')
            spec['_scope_protocol']=protocol
        return protocol


    def _captured(spec,snapshot,register=False):
        return _capture_protocol(spec).require('captured','register' if register else 'compute',
            payload=id(snapshot),validated=True,register=register)


    def _cancel_cycle(spec,future):
        protocol=spec.get('_scope_protocol')
        if protocol is None or protocol.event('cancel').action=='cancel':
            future.cancel()


    def _finish_cycle(spec,failed=False):
        """Release this actor only after its numerical helper has retired."""
        protocol=spec.get('_scope_protocol')
        if protocol is None:return
        phase=protocol.state.phase
        if phase in ('capturing','registering') or failed:protocol.event('failed')
        elif phase=='running':protocol.event('complete')
        if protocol.state.phase=='retiring':protocol.event('retired',validated=True)
        protocol.event('close')
        if protocol.state.phase=='retiring':protocol.event('retired',validated=True)
        protocol.close()
        spec.pop('_scope_protocol',None)


    def drain_retired():
        """Consume physical terminal receipts while actor cleanup is still open."""
        sampling=_application.component('scopes.sampling')
        scheduler=_application.component('graph.scheduler')
        for spec in tuple(_retiring_captures):
            if sampling.acquisition_retired(spec):
                scheduler.scope_complete(spec['_schedule'])
                _finish_cycle(spec);sampling.release(spec)
                _retiring_captures[:]=[item for item in _retiring_captures if item is not spec]
                kind,pointer=spec['_schedule'].channel
                states=_linear_states if kind=='linear' else _states
                if (pointer in states and states[pointer].get('requested')
                        and not _application.app.fenced and not _application.closing):
                    _wake(linear_tick if kind=='linear' else tick,.001)
        for spec,future in tuple(_retiring_cycles):
            if future.retired():
                _finish_cycle(spec);sampling.release(spec)
                _retiring_cycles[:]=[(s,f) for s,f in _retiring_cycles if f is not future]
        return not (_retiring_cycles or _retiring_captures)


    def _retirement_tick():
        _poll()
        return .01 if _retiring_cycles or _retiring_captures else None


    def _recovery_pending():
        # Automatic scope timers and direct UI requests share the undo barrier.
        nodes = _application.component('graph.nodes')
        return nodes.recovery_pending()


    class ScopeResult:
        """Decode a helper result once, retaining the numerical payload ownership."""
        def __init__(self, handle):
            self.handle = handle
            self._decoded = None
            self._preview_seen = None

        def done(self):
            return self.handle.done()

        def cancel(self):
            return self.handle.cancel()

        def retired(self):
            retired=getattr(self.handle,'retired',None)
            return retired() if retired is not None else self.handle.done()

        def preview(self):
            value=getattr(self.handle,'preview',None)
            if value is None or value is self._preview_seen:return None
            self._preview_seen=value
            return thaw_scope_result(value)

        def result(self, timeout=None):
            if self._decoded is None:
                self._decoded = thaw_scope_result(self.handle.result(timeout=timeout))
            return self._decoded


    def _submit(operation, snapshot):
        get_runtime = _application.component('jobs').get_runtime
        from ...blender_runtime.execution.contracts import JobSpec
        # Scope coordinators own source freshness and manual retries. A successful
        # helper result may still fail provenance validation, so it must not become
        # a runtime cache hit for the next capture with the same requested key.
        if operation == 'scope_full' and '_warp_views' in snapshot:
            operation = 'scope_warp'
        clouds = _application.component('scopes.clouds')
        active=operation in ('scope_register','scope_update')
        spec = JobSpec(operation, freeze_scope_request(snapshot),
                       lane=('scope_gpu' if clouds.enabled() else 'bulk') if active else
                            'linear' if operation == 'scope_linear' else 'gpu' if operation == 'scope_warp' else 'bulk',
                       priority=1 if active or operation == 'scope_warp' else 0)
        return ScopeResult(get_runtime().submit(spec))


    def _poll():
        capture=_application.components.get('scopes.capture')
        if capture is not None:capture.poll()
        if _running is not None or _linear_running is not None or _retiring_cycles or _retiring_captures:
            runtime=_application.component('services').current_runtime()
            if runtime is not None:runtime.poll(budget_ms=4)
        drain_retired()


    def _reuse_linear_sources(result):
        """Keep foreground source identities when a helper reuses their revision."""
        sources = []
        for name in ('_source', '_comparison_source'):
            source = result[name]
            cached = None if source is None else _linear_sources.get(source['key'])
            if cached is not None and cached['_scope_provenance'] == source['_scope_provenance']:
                source = cached
            sources.append(source)
        source, comparison = sources
        if source is result['_source'] and comparison is result['_comparison_source']:
            return result
        # Both the decoded envelope and cache entries are already deeply frozen.
        return MappingProxyType(dict(result, _source=source, _comparison_source=comparison,
                                     source_rgb=source['rgb'], alpha=source['alpha'],
                                     _comparison=comparison))


    def visible_consumers():
        diagram = _application.component('ui.diagram')
        return diagram.scope_consumers()


    def _schedule(node,kind,spec):
        """Attach an immutable coordinator token to this exact scope request."""
        scheduler = _application.component('graph.scheduler')
        states=_linear_states if kind=='linear' else _states
        state=states.get(spec.get('_scope_channel',node.as_pointer()),{})
        frozen=state.get('capturing')
        admitted=spec
        if (frozen is not None and '_scope_protocol' not in spec and
                spec['context']!=frozen['context'] and _compatible(frozen,spec)):
            # A partial acquisition can reveal its primary revision before its
            # reference retires. Preserve the active actor's lifetime until the
            # whole acquisition can bind its verified identities atomically.
            admitted=dict(spec)
            for name in ('context','scope_source','_scope_display_source','_scope_compatibility'):
                if name in frozen:admitted[name]=frozen[name]
        spec['_schedule']=scheduler.scope_request(node,kind,admitted)
        spec['_scope_asynchronous']=True
        return spec


    def _wake(fn,delay):
        scheduler = _application.component('graph.scheduler')
        scheduler.wake(fn,delay)


    def _unwake(fn):
        scheduler = _application.component('graph.scheduler')
        scheduler.unwake(fn)


    def _sync_consumers(throttle=False):
        nonlocal _consumers, _linear_pending, _consumer_scan_due
        now=time.monotonic()
        if throttle and now<_consumer_scan_due:return None
        _consumer_scan_due=now+.016
        latest=visible_consumers()
        added=set(latest)-set(_consumers)
        latest={key:tuple(value)+(None,)*(4-len(value)) for key,value in latest.items()}
        _consumers={key:(retain(node,'scope.consumer'),space,view,capture_context)
                    for key,(node,space,view,capture_context) in latest.items()}
        for kind,states in (('full',_states),('linear',_linear_states)):
            for pointer,state in states.items():
                if (kind,pointer) not in latest and not state.get('manual'):
                    state['requested']=False;state['probe']=False
                    _release_capture(state)
                elif (kind,pointer) in added and state['completed'] is not None:
                    # A hidden source may have changed since its last publication.
                    state['probe']=True;state['force_probe']=True
        if _linear_pending is not None:
            pointer,state,_=_linear_pending
            if not _wanted('linear',pointer,state):_linear_pending=None
        return latest


    def _wanted(kind,pointer,state):
        return bool(state.get('manual') or (kind,pointer) in _consumers)


    def _service_order(kind,states):
        """Stable resident order; UI refreshes only replace each pending payload."""
        order=_service[kind]
        next_pointer=order[_service_start[kind] % len(order)] if order else None
        order[:]=[pointer for pointer in order if pointer in states]
        order.extend(pointer for pointer in states if pointer not in order)
        _service_start[kind]=order.index(next_pointer) if next_pointer in order else 0
        return order


    def _runnable(kind,pointer,state):
        automatic=_application.component('scopes.automatic')
        return bool(automatic.analysis_allowed(resolve(state['spec']['node']),state['spec'].get('_capture_context')) and
                    state.get('requested') and _wanted(kind,pointer,state) and
                    not any(spec['_schedule'].channel==(kind,pointer) for spec in _retiring_captures) and
                    (state.get('probe') or state['spec']['key'] not in
                     (state['completed'],state['failed'])))


    def _candidates(kind,states):
        from ...perceptual_grade_core.graph import scheduler_policy as policy
        order=tuple(_service_order(kind,states))
        eligible=[_runnable(kind,pointer,states[pointer]) for pointer in order]
        while True:
            index=policy.scope_next(eligible,_service_start[kind])
            if index<0:return
            eligible[index]=False
            pointer=order[index]
            yield pointer,states[pointer]


    def _serviced(kind,pointer):
        order=_service[kind]
        if pointer in order:_service_start[kind]=(order.index(pointer)+1)%len(order)


    def _compatible(spec,latest):
        """Numeric edits may lag; source, tone/page and raster geometry may not."""
        sampling=_application.component('scopes.sampling')
        spec=sampling.bind_sources(spec);latest=sampling.bind_sources(latest)
        return (spec.get('_scope_epoch')==latest.get('_scope_epoch') and
                spec['context']==latest['context'] and
                spec.get('scope_source')==latest.get('scope_source') and
                spec.get('view')==latest.get('view') and
                spec.get('_scope_demands')==latest.get('_scope_demands') and
                spec.get('_scope_backend')==latest.get('_scope_backend') and
                tuple(v['key'] for v in spec.get('_warp_views',()))==
                tuple(v['key'] for v in latest.get('_warp_views',())))


    def _cycle(pointer,state,spec,latest,**phase):
        scheduler = _application.component('graph.scheduler')
        from ...perceptual_grade_core.graph import scheduler_policy as policy
        node=resolve(spec['node'])
        compatible=bool(latest is state and _compatible(spec,latest['spec']) and
                        scheduler.scope_coherent(node,'full',spec))
        return policy.scope_cycle(_wanted('full',pointer,state),compatible,
            bool(latest is state and latest['spec']['key']==spec['key']),
            scheduler.scope_dragging(node),history=bool(spec.get('_scope_history')),**phase)


    def _acknowledged(future,snapshot,result):
        from ...perceptual_grade_core.graph import scheduler_policy as policy
        handle=getattr(future,'handle',None)
        runtime=getattr(handle,'runtime',None)
        # Runtime messages are already attempt-fenced; also validate the seed's
        # declared identity before caching it or continuing a phased capture.
        return policy.scope_ack_admissible(True,result.get('cache_id')==snapshot['cache_id'],
            result.get('backend')==getattr(handle,'backend',result.get('backend')),
            result.get('session')==getattr(getattr(runtime,'session',None),'identity',result.get('session')),
            result.get('generation')==getattr(handle,'generation',result.get('generation')))


    def _new_state(spec,previous=None):
        previous=previous or {}
        _release_capture(previous)
        return dict(spec=spec,result=None,completed=None,failed=None,error='',capturing=None,
            requested=previous.get('requested',False),manual=previous.get('manual',False),
            probe=previous.get('probe',not previous),force_probe=previous.get('force_probe',False))


    def _same_context(state,spec):
        return state is not None and (state['spec']['context']==spec['context'] or
            (state.get('capturing') is not None and _compatible(state['capturing'],spec)))


    def _requested(state,ui):
        sampling = _application.component('scopes.sampling')
        spec=state['spec']
        if spec.get('_scope_unread'):state['probe']=True
        due=(sampling.probe_delay(resolve(spec['node']))
             if state['completed']==spec['key']
                and state['failed']!=spec['key'] else None)
        if due is not None and due<=0:state['probe']=True
        work=state['probe'] or spec['key'] not in (state['completed'],state['failed'])
        state['requested']=work
        if not ui and work:state['manual']=True
        return work


    def _current_epoch(state,spec):
        return (spec['key']==state['spec']['key'] and
                spec['_schedule'].epoch==state['spec']['_schedule'].epoch)

    def _completed(state,key,epoch=None):
        if key==state['spec']['key'] and (epoch is None or epoch==state['spec']['_schedule'].epoch):
            state['requested']=False;state['manual']=False


    def _probe(state,spec=None):
        """Pin acquisition across demand changes, then bind verified revisions."""
        if spec is None:spec=state['spec']
        if not state.get('probe') and not spec.get('_scope_unread'):return spec
        sampling = _application.component('scopes.sampling')
        prepared=_capture_effect(spec,lambda:sampling.prepare_sources(spec,
            force=state.get('force_probe',False),asynchronous=spec.get('_scope_asynchronous',False)))
        state['probe']=False;state['force_probe']=False
        if prepared['key']==spec['key']:return spec
        scheduler=_application.component('graph.scheduler')
        node=resolve(spec['node']);kind='linear' if spec.get('scope_stage')=='LINEAR' else 'full'
        latest=_describe(node,spec['space'],spec.get('view',()),kind=kind,
                         capture_context=spec.get('_capture_context'))
        if not _compatible(prepared,latest):
            raise ValueError('Scope source changed during acquisition')
        # The decode has physically retired. Rebind its now-known revision;
        # the newer numerical demand remains the channel's requested identity.
        scheduler.scope_complete(spec['_schedule'])
        prepared=_schedule(node,kind,prepared)
        if not scheduler.scope_start(prepared['_schedule']):
            raise RuntimeError('Scope acquisition could not bind its verified revision')
        state['capturing']=prepared
        state['spec']=_schedule(node,kind,latest)
        return prepared


    def _probe_delay():
        sampling = _application.component('scopes.sampling')
        values=[]
        for (kind,pointer),(node_ref,space,view,capture_context) in _consumers.items():
            node=resolve(node_ref)
            if node is None:continue
            state=(_linear_states if kind=='linear' else _states).get(pointer)
            if state is not None and state['failed']==state['spec']['key']:continue
            try:delay=.016 if state is not None and state['spec'].get('_scope_history') else sampling.probe_delay(node)
            except (ReferenceError,RuntimeError,ValueError):continue
            if delay is not None:values.append(delay)
        return min(values) if values else None


    def refresh_visible():
        """Discover visible source demands on an ordinary foreground turn."""
        if _recovery_pending():return
        automatic=_application.component('scopes.automatic')
        consumers=tuple(_sync_consumers().items())
        automatic.forget_except({pointer for (kind,pointer),_ in consumers})
        if not automatic.sync():
            if consumers:_wake(probe_tick,.05)
            return
        observed=set()
        for (kind,pointer),(node,space,view,capture_context) in consumers:
            try:
                if pointer not in observed:
                    automatic.observe(node,capture_context);observed.add(pointer)
                if not automatic.analysis_allowed(node,capture_context):continue
                if kind=='linear':request_linear(node,space,ui=True,sync=False,capture_context=capture_context)
                else:request(node,space,view,ui=True,sync=False,capture_context=capture_context)
            except (ValueError,ReferenceError,RuntimeError,TypeError) as error:
                if not _preparing(error):_request_errors[kind,pointer]=text(error)
        delay=_probe_delay()
        if delay is None and not consumers:_unwake(probe_tick)
        else:_wake(probe_tick,.05 if delay is None else max(.016,delay))

    def probe_tick():
        if _recovery_pending():return None
        refresh_visible()
        delay=_probe_delay()
        return max(.016,delay) if delay is not None else .05 if _consumers else None

    def _describe(node,space,view=(),kind='full',capture_context=None):
        """Expose pre-capture failures even when no scheduler ticket exists yet."""
        sampling = _application.component('scopes.sampling')
        pointer=sampling.channel(node,capture_context)
        try:
            kwargs={} if capture_context is None else {'capture_context':capture_context}
            spec=(sampling.describe_linear(node,space,**kwargs) if kind=='linear' else
                  sampling.describe(node,space,view,**kwargs))
        except (ValueError,ReferenceError,RuntimeError,TypeError,OSError,ArithmeticError) as error:
            input_state=sampling.scope_input_status(node,capture_context)
            if _preparing(error) or (input_state.get('native') and input_state.get('state') in {'waiting','capturing','updating','paused'}):
                _request_errors.pop((kind,pointer),None)
            else:
                _request_errors[kind,pointer]=text(error)
                _request_errors.move_to_end((kind,pointer))
            while len(_request_errors)>32:_request_errors.popitem(last=False)
            raise
        _request_errors.pop((kind,pointer),None)
        _request_errors.pop((kind,spec.get('_scope_channel',pointer)),None)
        _application.component('scopes.automatic').stamp(spec)
        return spec


    def _consumer_context(kind,node):
        sampling = _application.component('scopes.sampling')
        channel=sampling.channel(node)
        value=_consumers.get((kind,channel))
        return None if value is None else value[3]


    def request(node,space,view=(),ui=False,sync=True,capture_context=None):
        if _recovery_pending():return None
        if ui and sync:
            _wake(probe_tick,.001)
            return _application.component('graph.scheduler').scope_result(node,'full',channel=_application.component('scopes.sampling').channel(node,capture_context))
        sampling = _application.component('scopes.sampling')
        if ui and sync:_sync_consumers()
        if capture_context is None:capture_context=_consumer_context('full',node)
        spec=_schedule(node,'full',_describe(node,space,view,capture_context=capture_context))
        pointer=spec.get('_scope_channel',node.as_pointer());state=_states.get(pointer)
        if not _same_context(state,spec):
            state=_new_state(spec,state)
            _states[pointer]=state
        else:state['spec']=spec
        _states.move_to_end(pointer)
        _trim_states('full',_states,pointer)
        _service_order('full',_states)
        if _requested(state,ui) and _wanted('full',pointer,state):
            _wake(tick,.001)
        scheduler = _application.component('graph.scheduler')
        return scheduler.scope_result(node,'full',channel=pointer)


    def status(node,kind='full',channel=None):
        """Display availability is independent of an in-flight worker's state."""
        pointer=channel or _application.component('scopes.sampling').channel(node)
        state=(_linear_states if kind=='linear' else _states).get(pointer)
        scheduler = _application.component('graph.scheduler')
        available=scheduler.scope_result(node,kind,channel=pointer) is not None
        context=state['spec'].get('_capture_context') if state is not None else None
        input_state=_application.component('scopes.sampling').scope_input_status(node,context)
        if input_state.get('native') and input_state.get('state') in {'waiting','capturing','stale','updating','paused','error'}:
            return dict(state=input_state['state'],error=input_state.get('error',''),available=available,
                        frames=state.get('_displayed_frames',()) if state else (),
                        scene_frame=state.get('_displayed_scene_frame') if state else None)
        error=_request_errors.get((kind,pointer))
        if error:return dict(state='error',error=error,available=available)
        if state is None:return dict(state='preparing',error='',available=available)
        phase=scheduler.scope_status(node,kind,channel=pointer)
        failed=state['failed']==state['spec']['key']
        pending=state.get('_visual_pending',False) or state['completed']!=state['spec']['key']
        return dict(state='error' if failed or phase=='error' else 'current' if phase=='current' and not pending
            else 'updating' if available else 'preparing',error=state.get('error',''),available=available,
            frames=state.get('_displayed_frames',()),scene_frame=state.get('_displayed_scene_frame'))


    def retry(node,kind=None,channel=None):
        _application.component('scopes.automatic').retry(node,_consumer_context(kind or 'full',node))
        """Explicitly re-arm a failed request without discarding its last result."""
        scheduler = _application.component('graph.scheduler'); lut = _application.component('graph.lut')
        retried=[]
        pointer=channel or _application.component('scopes.sampling').channel(node)
        for lane,states in (('full',_states),('linear',_linear_states)):
            if kind is not None and kind!=lane:continue
            early=_request_errors.pop((lane,pointer),None)
            state=states.get(pointer)
            if state is None:continue
            if not early and state['failed']!=state['spec']['key'] and scheduler.scope_status(node,lane,channel=pointer)!='error':continue
            if not early and not scheduler.scope_retry(node,lane,channel=pointer):continue
            retried.append((lane,state))
        if not retried:return
        lut.scope_retry()
        for lane,state in retried:
            state['failed']=None;state['completed']=None;state['error']=''
            state['requested']=True;state['manual']=True
            state['probe']=True;state['force_probe']=True
            callback=linear_tick if lane=='linear' else tick
            _wake(callback,.001)


    def _failure(state,spec,error):
        scheduler = _application.component('graph.scheduler')
        if _current_epoch(state,spec):
            state['failed']=spec['key'];state['error']=text(error)
        protocol=spec.get('_scope_protocol')
        if protocol is not None:protocol.event('failed')
        if _current_epoch(state,spec):scheduler.scope_failed(spec['_schedule'],error)


    def _preparing(error):
        lut = _application.component('graph.lut')
        guard = _application.component('graph.output_guard')
        images=_application.component('scopes.images')
        return isinstance(error,(lut.ScopePreparing,guard.PublicationDeferred,images.SourcePreparing)) or text(error)=='Preparing compositor scopes'


    def _capture_effect(spec,action):
        """Waiting for a final dependency is a successful capture attempt, not failure."""
        scheduler = _application.component('graph.scheduler')
        ScopePreparing = _application.component('graph.lut').ScopePreparing
        pending=[]
        def attempt():
            try:return action()
            except (ValueError,ReferenceError,RuntimeError,OSError,ArithmeticError) as exc:
                if not _preparing(exc):raise
                # A traceback points back to this frame and its capture closure.
                # Keeping the exception here would retain node RNA in a cycle
                # after scope cleanup, until unrelated GC after native undo.
                pending.append(text(exc))
        result=scheduler.effect(spec['_schedule'],'capture',attempt)
        if pending:raise ScopePreparing(pending[0])
        return result


    def _release_capture(state):
        spec=state.get('capturing')
        if spec is None:return
        scheduler = _application.component('graph.scheduler')
        sampling = _application.component('scopes.sampling')
        state['capturing']=None
        if sampling.acquisition_retired(spec):
            scheduler.scope_complete(spec['_schedule']);_finish_cycle(spec);sampling.release(spec)
        else:
            _capture_protocol(spec).event('cancel')
            _retiring_captures.append(spec)
            if not _application.app.fenced and not _application.closing:
                _wake(_retirement_tick,.001)


    def finish_edit(node):
        """Queue the released settings without waiting for their scope analysis."""
        refresh_linear(node)
        for pointer,state in tuple(_states.items()):
            if resolve(state['spec']['node'])!=node or ('full',pointer) not in _consumers:continue
            try:request(node,state['spec']['space'],state['spec']['view'],ui=True,
                        capture_context=state['spec'].get('_capture_context'))
            except (ValueError,ReferenceError,RuntimeError):pass


    def refresh_backend():
        """Queue a new helper capture while keeping source-compatible displays."""
        for state in tuple(_states.values()):
            spec=state['spec'];node=resolve(spec['node'])
            if node is not None:
                request(node,spec['space'],spec['view'],ui=not state.get('manual'),sync=False,
                        capture_context=spec.get('_capture_context'))


    def _trim_states(kind,states,current):
        scheduler = _application.component('graph.scheduler')
        while len(states)>8:
            # Visible editors pin their slots; evict an unobserved owner first.
            victim=next((p for p in states if p!=current and (kind,p) not in _consumers),None)
            if victim is None:break
            _release_capture(states.pop(victim))
            scheduler.scope_forget(kind,victim)


    def request_linear(node,space,ui=False,sync=True,capture_context=None):
        """Keep one running calculation and replace the next requested matrix."""
        nonlocal _linear_pending
        if _recovery_pending():return None
        if ui and sync:
            _wake(probe_tick,.001)
            return _application.component('graph.scheduler').scope_result(node,'linear',channel=_application.component('scopes.sampling').channel(node,capture_context))
        sampling = _application.component('scopes.sampling')
        if ui and sync:_sync_consumers()
        if capture_context is None:capture_context=_consumer_context('linear',node)
        spec=_schedule(node,'linear',_describe(node,space,kind='linear',capture_context=capture_context))
        pointer=spec.get('_scope_channel',node.as_pointer())
        state=_linear_states.get(pointer)
        if not _same_context(state,spec):
            state=_new_state(spec,state)
            _linear_states[pointer]=state
        else:state['spec']=spec
        _linear_states.move_to_end(pointer)
        _trim_states('linear',_linear_states,pointer)
        _service_order('linear',_linear_states)
        _requested(state,ui)
        _choose_linear()
        if linear_pending():_wake(linear_tick,.001)
        scheduler = _application.component('graph.scheduler')
        return scheduler.scope_result(node,'linear',channel=pointer)


    def refresh_linear(node):
        """Refresh a visible regular-Grade scope before scheduling preview work."""
        for pointer,state in tuple(_linear_states.items()):
            if resolve(state['spec']['node'])!=node or ('linear',pointer) not in _consumers:continue
            try:request_linear(node,state['spec']['space'],ui=True,sync=False,
                               capture_context=state['spec'].get('_capture_context'))
            except (ValueError,ReferenceError,RuntimeError):pass


    def linear_pending():
        return _linear_running is not None or _linear_pending is not None


    def _choose_linear():
        nonlocal _linear_pending
        _linear_pending=None
        for pointer,state in _candidates('linear',_linear_states):
            if (_linear_running is not None and _linear_running[1] is state and
                    _linear_running[2]['key']==state['spec']['key']):continue
            _linear_pending=(pointer,state,state['spec']);break


    def _linear_latest(pointer,state,refresh=False):
        sampling = _application.component('scopes.sampling')
        if _linear_states.get(pointer) is not state:return None
        spec=state['spec']
        try:
            if not sampling.live(resolve(spec['node'])):raise ReferenceError('Removed scope node')
            if not refresh:return state
            latest=_schedule(resolve(spec['node']),'linear',_describe(resolve(spec['node']),spec['space'],kind='linear',
                capture_context=_consumers.get(('linear',pointer),(None,None,None,spec.get('_capture_context')))[3]))
        except (ValueError,ReferenceError,RuntimeError):
            _release_capture(state);_linear_states.pop(pointer,None);return None
        if not _same_context(state,latest):
            state=_new_state(latest,state)
            _linear_states[pointer]=state
        else:state['spec']=latest
        return state


    def linear_tick():
        """Run the priority matrix lane; the coordinator admits publications."""
        nonlocal _linear_running, _linear_pending
        _poll()
        if not _application.component('scopes.automatic').sync():return .05
        if _recovery_pending():return None
        scheduler = _application.component('graph.scheduler')
        from ...perceptual_grade_core.graph import scheduler_policy as policy
        _sync_consumers()
        if _linear_running is not None:
            pointer,state,spec,future=_linear_running
            latest=_linear_latest(pointer,state,refresh=future.done() and ('linear',pointer) in _consumers)
            compatible=bool(latest is state and _compatible(spec,latest['spec']) and
                            scheduler.scope_coherent(resolve(spec['node']),'linear',spec))
            decision=policy.scope_cycle(_wanted('linear',pointer,state),compatible,
                bool(latest is state and spec['key']==latest['spec']['key']),
                scheduler.scope_dragging(resolve(spec['node'])),history=bool(spec.get('_scope_history')))
            if decision.cancel:_cancel_cycle(spec,future)
            if not future.retired():return .001
            _linear_running=None
            token=spec['_schedule'];scheduler.scope_complete(token)
            if decision.keep:
                published=False
                try:
                    result=_reuse_linear_sources(scheduler.effect(token,'worker',future.result))
                    scheduler.scope_validate_payload(token,result)
                    _capture_protocol(spec).require('visual','publish_visual',payload=id(result),validated=True)
                    if _application.component('scopes.automatic').current(spec) and scheduler.scope_publish(token,result):
                        state['result']=result
                        if _current_epoch(state,spec):state['completed']=spec['key']
                        if _current_epoch(state,spec):state['failed']=None
                        state['_displayed_frames']=spec.get('_scope_frames',())
                        state['_displayed_scene_frame']=spec.get('_scope_scene_frame')
                        published=True
                        for source in (result['_source'],result['_comparison_source']):
                            if source is None:continue
                            _linear_sources[source['key']]=source;_linear_sources.move_to_end(source['key'])
                        while len(_linear_sources)>8:_linear_sources.popitem(last=False)
                except Exception as exc:
                    _failure(state,spec,exc);log_exception(LOG,'Perceptual Grade Stage-1 scope analysis failed')
                _completed(state,spec['key'],spec['_schedule'].epoch)
                if published:_redraw('linear',pointer)
            sampling = _application.component('scopes.sampling')
            _finish_cycle(spec)
            sampling.release(spec)
        _choose_linear()
        waiting=False
        for pointer,state in _candidates('linear',_linear_states):
            spec=state['spec']
            try:
                # Freeze immediately before capture; retain frozen dependencies
                # across numeric edits while another visible lane can make progress.
                state=_linear_latest(pointer,state,refresh=True)
                if state is None:continue
                spec=state['spec']
                if not state.get('probe') and spec['key'] in (state['completed'],state['failed']):
                    _completed(state,spec['key'],spec['_schedule'].epoch);continue
                frozen=state.get('capturing')
                if frozen is not None and (not _compatible(frozen,spec) or
                        (frozen['key']!=spec['key'] and not scheduler.scope_dragging(resolve(spec['node'])) and not frozen.get('_scope_history'))):
                    _release_capture(state);frozen=None
                if frozen is None:
                    if not _runnable('linear',pointer,state):waiting=True;continue
                    if not scheduler.scope_start(spec['_schedule']):waiting=True;continue
                    state['capturing']=spec
                    _capture_protocol(spec)
                else:spec=frozen
                spec=_probe(state,spec)
                if spec['key'] in (state['completed'],state['failed']):
                    _release_capture(state);_completed(state,spec['key'],spec['_schedule'].epoch);continue
                if not scheduler.scope_coherent(resolve(spec['node']),'linear',spec):
                    waiting=True;continue
                sampling = _application.component('scopes.sampling')
                snapshot=_capture_effect(spec,lambda:sampling.capture_linear(spec,_linear_sources))
                newest=_describe(resolve(spec['node']),spec['space'],kind='linear',capture_context=spec.get('_capture_context'))
                if not _compatible(spec,newest):
                    _release_capture(state);waiting=True;continue
            except Exception as exc:
                if _preparing(exc):waiting=True;continue
                _release_capture(state);_failure(state,spec,exc)
                log_exception(LOG,'Perceptual Grade Stage-1 scope capture failed')
                _completed(state,spec['key'],spec['_schedule'].epoch);_redraw('linear',pointer);continue
            try:
                _captured(spec,snapshot)
                future=scheduler.effect(spec['_schedule'],'submit',lambda:_submit('scope_linear',snapshot))
            except Exception as exc:
                _release_capture(state);_failure(state,spec,exc)
                _completed(state,spec['key'],spec['_schedule'].epoch);continue
            state['capturing']=None
            _linear_running=(pointer,state,spec,future)
            _serviced('linear',pointer);_choose_linear()
            return .001
        _choose_linear()
        return .016 if waiting else None


    def _latest(pointer,state,refresh=False,describe=True):
        sampling = _application.component('scopes.sampling')
        spec=state['spec']
        if not describe:return state
        if not sampling.live(resolve(spec['node'])):
            _release_capture(state);_states.pop(pointer,None);return None
        if not refresh:return state
        try:new=_schedule(resolve(spec['node']),'full',_describe(resolve(spec['node']),spec['space'],spec['view'],
            capture_context=_consumers.get(('full',pointer),(None,None,None,spec.get('_capture_context')))[3]))
        except (ValueError,ReferenceError,RuntimeError):
            _release_capture(state);_states.pop(pointer,None);return None
        if not _same_context(state,new):
            state=_new_state(new,state)
            _states[pointer]=state
        else:state['spec']=new
        return state


    def _redraw(kind,pointer):
        if (kind,pointer) in _consumers:
            lut = _application.component('graph.lut')
            lut.redraw()


    def _seed(snapshot):
        """Retain one immutable foreground source for normal cache-miss recovery."""
        cache_id=snapshot['cache_id']
        _scope_sources[cache_id]=snapshot;_scope_sources.move_to_end(cache_id)
        while len(_scope_sources)>8:
            expired,_=_scope_sources.popitem(last=False);_scope_receipts.pop(expired,None)


    def _update_snapshot(snapshot):
        image_fields={'rgba','chain','comparison','signal','to_grade','sample_grid'}
        result={name:value for name,value in snapshot.items() if name not in image_fields}
        result['matrix']=tuple(tuple(float(value) for value in row) for row in snapshot['matrix'])
        return result


    def _submit_active(spec,snapshot,register=False):
        operation='scope_register' if register else 'scope_update'
        spec['_scope_operation']=operation
        spec['_scope_snapshot']=snapshot
        return _submit(operation,snapshot if register else _update_snapshot(snapshot))


    def _merge_statistics(visual,delta):
        """Statistics can amend only the exact helper visual that requested them."""
        if visual is None or delta.get('_scope_request')!=visual.get('_scope_request'):
            return None
        colors=dict(visual['_active_colors'])
        for key,value in colors.items():
            if not key.endswith(':warp'):continue
            update=delta.get('_active_statistics',{}).get(key.split(':',1)[0])
            if update is None:
                update=dict(neutral=None,statistics_status=delta.get('_scope_stats_status','unavailable'))
            colors[key]=MappingProxyType(dict(value,**update))
        result=dict(visual,_active_colors=MappingProxyType(colors),_scope_part='stats',
                    _active_statistics_pending=(),_scope_stats_status=delta.get('_scope_stats_status','ready'))
        if delta.get('_scope_stats_error'):
            result['_scope_stats_error']=delta['_scope_stats_error']
        result['_scope_timings']=MappingProxyType(dict(visual.get('_scope_timings',{}),**delta.get('_scope_timings',{})))
        return MappingProxyType(result)


    def _publish(state,spec,payload,quality=2):
        if not _application.component('scopes.automatic').current(spec):return False
        scheduler = _application.component('graph.scheduler')
        token=replace(spec['_schedule'],quality=quality)
        scheduler.scope_validate_payload(token,payload)
        statistics=payload.get('_scope_part')=='stats'
        _capture_protocol(spec).require('statistics' if statistics else 'visual',
            'publish_statistics' if statistics else 'publish_visual',payload=id(payload),validated=True)
        if not scheduler.scope_publish(token,payload):return False
        state['result']=scheduler.scope_result(resolve(spec['node']),'full',channel=spec.get('_scope_channel'))
        if _current_epoch(state,spec):state['failed']=None
        state['_published_key']=spec['key'];state['_displayed_frames']=spec.get('_scope_frames',())
        state['_displayed_scene_frame']=spec.get('_scope_scene_frame')
        state['_visual_pending']=bool(quality==1)
        if quality==2 and _current_epoch(state,spec):state['completed']=spec['key']
        if payload.get('_scope_stats_error'):
            _failure(state,spec,payload['_scope_stats_error'])
        return True


    def _drain_preview(future):
        """Retirement polling may adopt a visual and terminal delta together."""
        preview=future.preview() if hasattr(future,'preview') else None
        retired=future.retired()
        if retired and hasattr(future,'preview'):
            final_preview=future.preview()
            if final_preview is not None:preview=final_preview
        return preview,retired


    def tick():
        """Complete one coherent visual/statistics cycle, then service newest work."""
        nonlocal _running
        _poll()
        if not _application.component('scopes.automatic').sync():return .05
        if _recovery_pending():return None
        scheduler = _application.component('graph.scheduler')
        sampling = _application.component('scopes.sampling')
        scanned=_sync_consumers(throttle=True)
        if _running is not None:
            pointer,state,spec,future=_running
            latest=_latest(pointer,state,describe=scanned is not None) if _states.get(pointer) is state else None
            decision=_cycle(pointer,state,spec,latest)
            # The helper must physically retire before releasing this capture's
            # scheduler ticket. A cancelled consumer is already done at runtime.
            if decision.cancel:_cancel_cycle(spec,future)
            try:
                preview,retired=_drain_preview(future)
                if (preview is not None or retired) and ('full',pointer) in _consumers:
                    # Throttle polling scans, never the validity check at an
                    # actual publication or physical worker retirement.
                    latest=_latest(pointer,state,refresh=True) if _states.get(pointer) is state else None
                decision=_cycle(pointer,state,spec,latest,retired=retired,preview=preview is not None)
                if decision.cancel:_cancel_cycle(spec,future)
                if decision.publish:
                    # Keep the visual belonging to this job even if publication
                    # was redundant; statistics must never join another raster.
                    spec['_scope_visual']=preview
                    if _publish(state,spec,preview,quality=1):_redraw('full',pointer)
            except Exception as exc:
                # Retirement may need more timer turns. Retain diagnostic text,
                # never the traceback and its references to the current RNA frame.
                spec['_scope_error']=text(exc)
                _cancel_cycle(spec,future)
                retired=future.retired()
            if not retired:return .001
            _running=None
            try:
                if decision.keep and '_scope_error' in spec:raise RuntimeError(spec.pop('_scope_error'))
                result=scheduler.effect(spec['_schedule'],'worker',future.result) if decision.collect else None
                control=result.get('_scope_control') if result is not None else None
                if control is not None:
                    snapshot=spec['_scope_snapshot'];cache_id=snapshot['cache_id']
                    acknowledged=_acknowledged(future,snapshot,result)
                    decision=_cycle(pointer,state,spec,latest,retired=True,control=True,ack=acknowledged)
                    if not decision.advance:raise ValueError('Scope source acknowledgement does not match its helper capture')
                    effect=_capture_protocol(spec).event(control,validated=True)
                    if effect.action != ('compute' if control=='ready' else 'register'):
                        raise ValueError('Scope coordinator rejected the captured source acknowledgement')
                    if control=='ready':
                        _scope_receipts[cache_id]=result
                        future=scheduler.effect(spec['_schedule'],'submit',lambda:_submit_active(spec,snapshot))
                    elif control=='needs_seed':
                        _scope_receipts.pop(cache_id,None)
                        future=scheduler.effect(spec['_schedule'],'submit',lambda:_submit_active(spec,snapshot,register=True))
                    else:raise ValueError('Unknown scope source acknowledgement')
                    _running=(pointer,state,spec,future)
                    return .001
                if result is not None:
                    if result.get('_scope_part')=='stats':result=_merge_statistics(spec.get('_scope_visual'),result)
                    if result is not None and _publish(state,spec,result):_redraw('full',pointer)
            except Exception as exc:
                if decision.keep:
                    _failure(state,spec,exc)
                    log_exception(LOG,'Perceptual Grade scope analysis failed')
            scheduler.scope_complete(spec['_schedule']);_finish_cycle(spec);sampling.release(spec)
            _completed(state,spec['key'],spec['_schedule'].epoch)
            decision=_cycle(pointer,state,spec,latest,retired=True,
                pending=any(_runnable('full',p,s) for p,s in _states.items()))
            if not decision.dispatch:return None
        waiting=False
        for pointer,state in _candidates('full',_states):
            state=_latest(pointer,state)
            if state is None:continue
            spec=state['spec']
            try:
                state=_latest(pointer,state,refresh=True)
                if state is None:continue
                spec=state['spec']
                if not state.get('probe') and spec['key'] in (state['completed'],state['failed']):
                    _completed(state,spec['key'],spec['_schedule'].epoch);continue
                frozen=state.get('capturing')
                if frozen is not None and (not _compatible(frozen,spec) or
                        (frozen['key']!=spec['key'] and not scheduler.scope_dragging(resolve(spec['node'])) and not frozen.get('_scope_history'))):
                    _release_capture(state);frozen=None
                if frozen is None:
                    if not _runnable('full',pointer,state):waiting=True;continue
                    if not scheduler.scope_start(spec['_schedule']):waiting=True;continue
                    state['capturing']=spec
                    _capture_protocol(spec)
                else:spec=frozen
                spec=_probe(state,spec)
                if spec['key'] in (state['completed'],state['failed']):
                    _release_capture(state);_completed(state,spec['key'],spec['_schedule'].epoch);continue
                if not scheduler.scope_coherent(resolve(spec['node']),'full',spec):
                    waiting=True;continue
                active='_scope_demands' in spec
                source=_scope_sources.get(sampling.cache_id(spec)) if active else None
                snapshot=_capture_effect(spec,lambda:sampling.capture(spec,seed=source) if active else sampling.capture(spec))
                newest=_describe(resolve(spec['node']),spec['space'],spec['view'],capture_context=spec.get('_capture_context'))
                if (newest['context']!=spec['context'] or newest['scope_source']!=spec['scope_source']
                        or not scheduler.scope_coherent(resolve(spec['node']),'full',spec)):
                    _release_capture(state);waiting=True;continue
                if active:
                    scheduler.scope_validate_payload(spec['_schedule'],snapshot)
                    _seed(snapshot)
            except Exception as exc:
                if _preparing(exc):waiting=True;continue
                _release_capture(state);_failure(state,spec,exc)
                log_exception(LOG,'Perceptual Grade scope capture failed')
                _completed(state,spec['key'],spec['_schedule'].epoch);_redraw('full',pointer);continue
            try:
                _captured(spec,snapshot,register=active and snapshot['cache_id'] not in _scope_receipts)
                future=scheduler.effect(spec['_schedule'],'submit',lambda:
                    _submit_active(spec,snapshot,register=snapshot['cache_id'] not in _scope_receipts)
                    if active else _submit('scope_full',snapshot))
            except Exception as exc:
                _release_capture(state);_failure(state,spec,exc)
                _completed(state,spec['key'],spec['_schedule'].epoch);continue
            state['capturing']=None
            _running=(pointer,state,spec,future)
            _serviced('full',pointer)
            return .001
        return .016 if waiting else None


    def clear():
        nonlocal _running, _linear_running, _linear_pending, _consumer_scan_due
        scheduler = _application.component('graph.scheduler')
        sampling = _application.component('scopes.sampling')
        for state in tuple(_states.values())+tuple(_linear_states.values()):_release_capture(state)
        for running in (_running,_linear_running):
            if running is not None:
                spec,future=running[2],running[3]
                _cancel_cycle(spec,future)
                if future.retired():_finish_cycle(spec);sampling.release(spec)
                else:_retiring_cycles.append((spec,future))
        scheduler.scope_clear()
        _unwake(probe_tick)
        _consumers.clear();_request_errors.clear()
        _unwake(tick)
        _states.clear()
        _running=None
        _unwake(linear_tick)
        _linear_states.clear();_linear_sources.clear();_linear_pending=None
        _scope_sources.clear();_scope_receipts.clear()
        for kind in _service:
            _service[kind].clear();_service_start[kind]=0
        _consumer_scan_due=0.
        _linear_running=None
        drain_retired()
        if (_retiring_cycles or _retiring_captures) and not _application.app.fenced and not _application.closing:
            _wake(_retirement_tick,.001)

    def _set_export_LOG(value):
        nonlocal LOG
        LOG = value

    def _set_export__consumer_scan_due(value):
        nonlocal _consumer_scan_due
        _consumer_scan_due = value

    def _set_export__consumers(value):
        nonlocal _consumers
        _consumers = value

    def _set_export__linear_pending(value):
        nonlocal _linear_pending
        _linear_pending = value

    def _set_export__linear_running(value):
        nonlocal _linear_running
        _linear_running = value

    def _set_export__linear_sources(value):
        nonlocal _linear_sources
        _linear_sources = value

    def _set_export__linear_states(value):
        nonlocal _linear_states
        _linear_states = value

    def _set_export__running(value):
        nonlocal _running
        _running = value

    def _set_export__scope_receipts(value):
        nonlocal _scope_receipts
        _scope_receipts = value

    def _set_export__scope_sources(value):
        nonlocal _scope_sources
        _scope_sources = value

    def _set_export__service(value):
        nonlocal _service
        _service = value

    def _set_export__service_start(value):
        nonlocal _service_start
        _service_start = value

    def _set_export__states(value):
        nonlocal _states
        _states = value

    def _set_export__recovery_pending(value):
        nonlocal _recovery_pending
        _recovery_pending = value

    def _set_export_ScopeResult(value):
        nonlocal ScopeResult
        ScopeResult = value

    def _set_export__submit(value):
        nonlocal _submit
        _submit = value

    def _set_export__poll(value):
        nonlocal _poll
        _poll = value

    def _set_export__reuse_linear_sources(value):
        nonlocal _reuse_linear_sources
        _reuse_linear_sources = value

    def _set_export__consumer_context(value):
        nonlocal _consumer_context
        _consumer_context = value

    def _set_export_visible_consumers(value):
        nonlocal visible_consumers
        visible_consumers = value

    def _set_export__schedule(value):
        nonlocal _schedule
        _schedule = value

    def _set_export__wake(value):
        nonlocal _wake
        _wake = value

    def _set_export__unwake(value):
        nonlocal _unwake
        _unwake = value

    def _set_export__sync_consumers(value):
        nonlocal _sync_consumers
        _sync_consumers = value

    def _set_export__wanted(value):
        nonlocal _wanted
        _wanted = value

    def _set_export__service_order(value):
        nonlocal _service_order
        _service_order = value

    def _set_export__runnable(value):
        nonlocal _runnable
        _runnable = value

    def _set_export__candidates(value):
        nonlocal _candidates
        _candidates = value

    def _set_export__serviced(value):
        nonlocal _serviced
        _serviced = value

    def _set_export__compatible(value):
        nonlocal _compatible
        _compatible = value

    def _set_export__cycle(value):
        nonlocal _cycle
        _cycle = value

    def _set_export__acknowledged(value):
        nonlocal _acknowledged
        _acknowledged = value

    def _set_export__new_state(value):
        nonlocal _new_state
        _new_state = value

    def _set_export__requested(value):
        nonlocal _requested
        _requested = value

    def _set_export__completed(value):
        nonlocal _completed
        _completed = value

    def _set_export__probe(value):
        nonlocal _probe
        _probe = value

    def _set_export__probe_delay(value):
        nonlocal _probe_delay
        _probe_delay = value

    def _set_export_refresh_visible(value):
        nonlocal refresh_visible
        refresh_visible = value

    def _set_export_probe_tick(value):
        nonlocal probe_tick
        probe_tick = value

    def _set_export_request(value):
        nonlocal request
        request = value

    def _set_export_status(value):
        nonlocal status
        status = value

    def _set_export_retry(value):
        nonlocal retry
        retry = value

    def _set_export__failure(value):
        nonlocal _failure
        _failure = value

    def _set_export__preparing(value):
        nonlocal _preparing
        _preparing = value

    def _set_export__capture_effect(value):
        nonlocal _capture_effect
        _capture_effect = value

    def _set_export__release_capture(value):
        nonlocal _release_capture
        _release_capture = value

    def _set_export_finish_edit(value):
        nonlocal finish_edit
        finish_edit = value

    def _set_export_refresh_backend(value):
        nonlocal refresh_backend
        refresh_backend = value

    def _set_export__trim_states(value):
        nonlocal _trim_states
        _trim_states = value

    def _set_export_request_linear(value):
        nonlocal request_linear
        request_linear = value

    def _set_export_refresh_linear(value):
        nonlocal refresh_linear
        refresh_linear = value

    def _set_export_linear_pending(value):
        nonlocal linear_pending
        linear_pending = value

    def _set_export__choose_linear(value):
        nonlocal _choose_linear
        _choose_linear = value

    def _set_export__linear_latest(value):
        nonlocal _linear_latest
        _linear_latest = value

    def _set_export_linear_tick(value):
        nonlocal linear_tick
        linear_tick = value

    def _set_export__latest(value):
        nonlocal _latest
        _latest = value

    def _set_export__redraw(value):
        nonlocal _redraw
        _redraw = value

    def _set_export__seed(value):
        nonlocal _seed
        _seed = value

    def _set_export__update_snapshot(value):
        nonlocal _update_snapshot
        _update_snapshot = value

    def _set_export__submit_active(value):
        nonlocal _submit_active
        _submit_active = value

    def _set_export__merge_statistics(value):
        nonlocal _merge_statistics
        _merge_statistics = value

    def _set_export__publish(value):
        nonlocal _publish
        _publish = value

    def _set_export__drain_preview(value):
        nonlocal _drain_preview
        _drain_preview = value

    def _set_export_tick(value):
        nonlocal tick
        tick = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        'drain_retired': lambda: drain_retired,
        '_retiring_cycles': lambda: _retiring_cycles,
        '_retiring_captures': lambda: _retiring_captures,
        "__name__": lambda: __name__,
        'LOG': lambda: LOG,
        'MappingProxyType': lambda: MappingProxyType,
        'OrderedDict': lambda: OrderedDict,
        'ScopeResult': lambda: ScopeResult,
        '_acknowledged': lambda: _acknowledged,
        '_candidates': lambda: _candidates,
        '_capture_effect': lambda: _capture_effect,
        '_choose_linear': lambda: _choose_linear,
        '_compatible': lambda: _compatible,
        '_completed': lambda: _completed,
        '_consumer_scan_due': lambda: _consumer_scan_due,
        '_consumers': lambda: _consumers,
        '_cycle': lambda: _cycle,
        '_drain_preview': lambda: _drain_preview,
        '_failure': lambda: _failure,
        '_latest': lambda: _latest,
        '_linear_latest': lambda: _linear_latest,
        '_linear_pending': lambda: _linear_pending,
        '_linear_running': lambda: _linear_running,
        '_linear_sources': lambda: _linear_sources,
        '_linear_states': lambda: _linear_states,
        '_merge_statistics': lambda: _merge_statistics,
        '_new_state': lambda: _new_state,
        '_poll': lambda: _poll,
        '_preparing': lambda: _preparing,
        '_probe': lambda: _probe,
        '_probe_delay': lambda: _probe_delay,
        '_publish': lambda: _publish,
        '_recovery_pending': lambda: _recovery_pending,
        '_redraw': lambda: _redraw,
        '_release_capture': lambda: _release_capture,
        '_requested': lambda: _requested,
        '_reuse_linear_sources': lambda: _reuse_linear_sources,
        '_runnable': lambda: _runnable,
        '_running': lambda: _running,
        '_schedule': lambda: _schedule,
        '_scope_receipts': lambda: _scope_receipts,
        '_scope_sources': lambda: _scope_sources,
        '_seed': lambda: _seed,
        '_service': lambda: _service,
        '_service_order': lambda: _service_order,
        '_service_start': lambda: _service_start,
        '_serviced': lambda: _serviced,
        '_states': lambda: _states,
        '_submit': lambda: _submit,
        '_submit_active': lambda: _submit_active,
        '_sync_consumers': lambda: _sync_consumers,
        '_trim_states': lambda: _trim_states,
        '_unwake': lambda: _unwake,
        '_update_snapshot': lambda: _update_snapshot,
        '_wake': lambda: _wake,
        '_wanted': lambda: _wanted,
        'clear': lambda: clear,
        'finish_edit': lambda: finish_edit,
        'freeze_scope_request': lambda: freeze_scope_request,
        'linear_pending': lambda: linear_pending,
        'linear_tick': lambda: linear_tick,
        'log_exception': lambda: log_exception,
        'logging': lambda: logging,
        'probe_tick': lambda: probe_tick,
        'readonly': lambda: readonly,
        'refresh_backend': lambda: refresh_backend,
        'refresh_linear': lambda: refresh_linear,
        'refresh_visible': lambda: refresh_visible,
        'replace': lambda: replace,
        'request': lambda: request,
        'request_linear': lambda: request_linear,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'retry': lambda: retry,
        'status': lambda: status,
        'text': lambda: text,
        'thaw_scope_result': lambda: thaw_scope_result,
        'tick': lambda: tick,
        'time': lambda: time,
        '_consumer_context': lambda: _consumer_context,
        'visible_consumers': lambda: visible_consumers,
    }, writers={
        'LOG': _set_export_LOG,
        '_consumer_scan_due': _set_export__consumer_scan_due,
        '_consumers': _set_export__consumers,
        '_linear_pending': _set_export__linear_pending,
        '_linear_running': _set_export__linear_running,
        '_linear_sources': _set_export__linear_sources,
        '_linear_states': _set_export__linear_states,
        '_running': _set_export__running,
        '_scope_receipts': _set_export__scope_receipts,
        '_scope_sources': _set_export__scope_sources,
        '_service': _set_export__service,
        '_service_start': _set_export__service_start,
        '_states': _set_export__states,
    '_recovery_pending': _set_export__recovery_pending, 'ScopeResult': _set_export_ScopeResult, '_submit': _set_export__submit, '_poll': _set_export__poll, '_reuse_linear_sources': _set_export__reuse_linear_sources, '_consumer_context': _set_export__consumer_context, 'visible_consumers': _set_export_visible_consumers, '_schedule': _set_export__schedule, '_wake': _set_export__wake, '_unwake': _set_export__unwake, '_sync_consumers': _set_export__sync_consumers, '_wanted': _set_export__wanted, '_service_order': _set_export__service_order, '_runnable': _set_export__runnable, '_candidates': _set_export__candidates, '_serviced': _set_export__serviced, '_compatible': _set_export__compatible, '_cycle': _set_export__cycle, '_acknowledged': _set_export__acknowledged, '_new_state': _set_export__new_state, '_requested': _set_export__requested, '_completed': _set_export__completed, '_probe': _set_export__probe, '_probe_delay': _set_export__probe_delay, 'refresh_visible': _set_export_refresh_visible, 'probe_tick': _set_export_probe_tick, 'request': _set_export_request, 'status': _set_export_status, 'retry': _set_export_retry, '_failure': _set_export__failure, '_preparing': _set_export__preparing, '_capture_effect': _set_export__capture_effect, '_release_capture': _set_export__release_capture, 'finish_edit': _set_export_finish_edit, 'refresh_backend': _set_export_refresh_backend, '_trim_states': _set_export__trim_states, 'request_linear': _set_export_request_linear, 'refresh_linear': _set_export_refresh_linear, 'linear_pending': _set_export_linear_pending, '_choose_linear': _set_export__choose_linear, '_linear_latest': _set_export__linear_latest, 'linear_tick': _set_export_linear_tick, '_latest': _set_export__latest, '_redraw': _set_export__redraw, '_seed': _set_export__seed, '_update_snapshot': _set_export__update_snapshot, '_submit_active': _set_export__submit_active, '_merge_statistics': _set_export__merge_statistics, '_publish': _set_export__publish, '_drain_preview': _set_export__drain_preview, 'tick': _set_export_tick, 'clear': _set_export_clear})
