"""Foreground automatic scope demand and playback publication fences.

Callbacks only latch an epoch. The source-bound actor owns one active request
and its newest successor; physical snapshot leases remain in scopes.capture.
"""
from ..integration.components import Component


def create(application):
    from itertools import count
    import bpy
    from ..runtime.protocol import Protocol
    ownership = application.component('ownership')
    records = {}
    serial = 1
    observed = 1
    playback = False
    sequence = count(1)

    def playback_pre(*_args):
        nonlocal serial, playback
        playback = True
        serial += 1

    def playback_post(*_args):
        nonlocal serial, playback
        playback = False
        serial += 1

    def playing():
        return playback or any(getattr(window.screen,'is_animation_playing',False)
            for manager in bpy.data.window_managers for window in manager.windows)

    def epoch():
        return serial

    def sync():
        nonlocal observed
        paused = playing()
        if observed != serial:
            observed = serial
            for record in tuple(records.values()):
                record['protocol'].event('pause')
                application.component('scopes.capture').cancel_scope(ownership.resolve(record['node']), record['context'])
                record['key'] = None
            observed = serial
        for record in tuple(records.values()):
            protocol = record['protocol']
            if paused and not protocol.state.paused:
                protocol.event('pause')
                application.component('scopes.capture').cancel_scope(ownership.resolve(record['node']), record['context'])
                record['key'] = None
            elif not paused and protocol.state.paused:
                protocol.event('resume')
        return not paused

    def register():
        application.add_handler('animation_playback_pre', playback_pre)
        application.add_handler('animation_playback_post', playback_post)

    def _address(node, context):
        return application.component('scopes.sampling').channel(node, context)

    def _complete(address, receipt, succeeded):
        record = records.get(address)
        if record is None or not sync():
            return False
        if succeeded:
            node=ownership.resolve(record['node'])
            if node is None:return False
            observe(node,record['context'])
        effect = record['protocol'].event('complete' if succeeded else 'failed',
            **receipt, validated=succeeded)
        return effect.action == 'publish'

    def observe(node, context=None):
        if not sync():
            return
        if not application.component('graph.output_guard').available():
            return
        sampling = application.component('scopes.sampling')
        owner, context, inputs = sampling.native_inputs(node, context)
        address = _address(node, context)
        if not inputs:
            record = records.pop(address, None)
            if record is not None:
                record['protocol'].event('close'); record['protocol'].close()
                application.component('scopes.capture').cancel_scope(ownership.resolve(record['node']), record['context'])
                ownership.release(record['node'])
            return
        record = records.get(address)
        if record is None:
            record = records[address] = dict(protocol=Protocol(application, 'scope_demand',initial={'generation':1}),
                node=ownership.retain(node, 'scope.automatic'), context=context, key=None, inputs=())
        protocol = record['protocol']
        key = tuple(value[1]['identity'] for value in inputs)
        continuity = tuple(value[1]['continuity'] for value in inputs)
        scheduler = application.component('graph.scheduler')
        continuity_key=scheduler.identity(continuity)
        if protocol.state.active and protocol.state.active_continuity != continuity_key:
            protocol.event('invalidate')
            application.component('scopes.capture').cancel_scope(owner,context)
            record['key']=None
        if key != record['key']:
            protocol.event('demand', epoch=protocol.state.epoch, request=next(sequence),
                source=scheduler.identity(key), continuity=continuity_key)
            record.update(key=key, inputs=inputs, context=context)
        if application.component('scopes.capture').busy(owner,context):return
        effect = protocol.event('start', epoch=protocol.state.epoch)
        if effect.action != 'capture':
            return
        receipt = dict(epoch=effect.epoch, request=effect.request, source=effect.source)
        frozen = tuple((socket, dict(descriptor, _scope_epoch=serial), size, exact)
                       for socket, descriptor, size, exact in record['inputs'])
        try:
            accepted = application.component('scopes.capture').acquire(owner, frozen,
                complete=lambda success: _complete(address, receipt, success))
            if not accepted:
                protocol.event('failed', **receipt)
            return accepted
        except Exception:
            protocol.event('failed', **receipt)
            raise

    def frame(context=None):
        name=context.get('scene') if context is not None else None
        scene=bpy.data.scenes.get(name) if name else bpy.context.scene
        return None if scene is None else (scene.name,int(scene.frame_current),float(scene.frame_subframe))

    def stamp(spec):
        spec['_scope_epoch']=serial
        spec['_scope_frame']=frame(spec.get('_capture_context'))
        return spec

    def current(request):
        return not playing() and request.get('_scope_epoch',serial)==serial

    def analysis_allowed(node, context=None):
        if not sync():
            return False
        capture = application.component('scopes.capture')
        record=records.get(_address(node,context))
        if record is None:return True
        if record['protocol'].state.accepted==0:return False
        for socket,_,_,_ in record['inputs']:
            descriptor = capture.accepted_descriptor(node, socket, context)
            if descriptor is not None and not current(descriptor):
                return False
        return True

    def status(node, context=None):
        if playing():
            return 'paused'
        record = records.get(_address(node, context))
        if record is None:
            return None
        state = record['protocol'].state
        if state.failed==state.desired and not state.active:return 'error'
        return 'updating' if state.active or state.desired > state.accepted else 'ready'

    def retry(node, context=None):
        record=records.get(_address(node,context))
        if record is not None:
            record['protocol'].event('retry',epoch=record['protocol'].state.epoch)
        observe(node,context)

    def forget_except(addresses):
        for address, record in tuple(records.items()):
            if address in addresses:
                continue
            records.pop(address)
            record['protocol'].event('close'); record['protocol'].close()
            application.component('scopes.capture').cancel_scope(ownership.resolve(record['node']), record['context'])
            ownership.release(record['node'])

    def clear():
        nonlocal serial, observed, playback
        forget_except(set())
        serial += 1; observed = serial; playback = False

    return Component(dict(__name__=__name__, register=register, observe=observe, sync=sync, playing=playing,
        epoch=epoch, stamp=stamp, current=current, analysis_allowed=analysis_allowed,
        status=status, retry=retry, forget_except=forget_except, clear=clear))
