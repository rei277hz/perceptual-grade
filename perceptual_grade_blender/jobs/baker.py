"""Foreground numerical-job adapters; no shader dispatch or LUT packing."""

from ..integration.components import Component

def create(_application):
    from concurrent.futures import CancelledError
    import time
    from ..runtime.protocol import Protocol

    from ...perceptual_grade_core.model import lut
    from ...perceptual_grade_core.jobs.codec import freeze_model, freeze_cube, thaw_cube
    from ...blender_runtime.execution.contracts import JobSpec
    get_runtime = _application.component('services').get_runtime


    class RemoteBaker:
        """Keep the coordinator's progressive Baker interface over a job handle."""
        requires_main_thread=False

        def __init__(self,model,progressive=True,encoded=False,stack=False,lane='bulk'):
            self.model=model;self.progressive=progressive;self.encoded=encoded;self.stack=stack
            self.space=lut.model_space(model);self.signature=lut.signature(model)
            self.cube=None;self.preview=None;self.handle=None;self.lane=lane
            self._inputs=freeze_model(model);self._preview_payload=None
            self._backend='GPU' if lane=='gpu' else 'CPU';self.device=''
            self._fallback_reason=None;self.packing=[]
            self._target=id(self)
            self.protocol=Protocol(_application, 'bake', identity={
                'generation': 1, 'revision': 1, 'target': self._target, 'mode': 'lut'})

        def _event(self, kind, **values):
            return self.protocol.event(kind, revision=1, target=self._target, **values)

        def _require(self, kind, action, **values):
            return self.protocol.require(kind, action, revision=1, target=self._target, **values)

        def _admit(self):
            if self.protocol.state.phase == 'idle':
                self._require('request', 'admit')
                self._require('admitted', 'compute', accepted=True)

        def _retire(self):
            if self.protocol.state.phase == 'retiring' and (self.handle is None or self.handle.retired()):
                self._event('retired', accepted=True)

        def _failed(self):
            self._event('failed')
            self.cancel()

        def _accept_preview(self, payload):
            if self.protocol.state.closing:
                raise CancelledError('Obsolete LUT preview')
            preview=thaw_cube(payload)
            if preview.size!=lut.PREVIEW_SIZE or preview.signature!=lut.signature(self.model,lut.PREVIEW_SIZE):
                raise ValueError('Helper returned an incompatible preview LUT')
            self._admit()
            self._require('preview', 'preview', candidate=1, accepted=True)
            self.preview=preview;self._preview_payload=payload

        def _accept_candidate(self, cube):
            if cube.size!=lut.SIZE or cube.signature!=self.signature:
                raise ValueError('Helper returned an incompatible final LUT')
            self._admit()
            if self.protocol.state.phase == 'computing':
                self._require('candidate', 'stage', candidate=2, accepted=True)
            elif self.protocol.state.phase not in ('staged', 'publishing', 'repair_needed', 'ready'):
                raise CancelledError('Obsolete LUT candidate')

        def prepare_publication(self):
            if self.cube is None or self.protocol.state.closing:return False
            self._accept_candidate(self.cube)
            return True

        def authorize_publication(self):
            if self.protocol.state.phase == 'ready':return True
            if self.protocol.state.phase == 'repair_needed':
                return self._require('retry', 'repair', candidate=2).accepted
            return self._require('publish', 'publish', candidate=2, accepted=True).accepted

        def published(self):
            if self.protocol.state.phase == 'ready':return True
            return self._event('published', candidate=2, accepted=True).accepted

        def publication_failed(self):
            return self._event('failed').accepted

        @property
        def backend(self):
            if self.cube is not None:return self.cube.diagnostics.get('backend',self._backend)
            if self.handle is not None:
                return 'GPU' if self.handle.backend=='gpu' else 'CPU'
            return self._backend

        @property
        def fallback_reason(self):
            return getattr(self.handle,'fallback_reason',None) or self._fallback_reason

        @fallback_reason.setter
        def fallback_reason(self,value):self._fallback_reason=value

        @property
        def progress(self):
            return 1. if self.cube is not None else (self.handle.progress if self.handle is not None else 0.)

        @property
        def cursor(self):return int(self.progress*lut.SIZE**3)

        @property
        def calculated(self):return self.cube is not None

        def start(self):
            if self.protocol.state.closing and self.handle is None:raise CancelledError('Obsolete LUT bake')
            if self.handle is None and self.cube is None:
                self._admit()
                try:
                    self.handle=get_runtime().submit(JobSpec('lut_bake',
                        dict(model=self._inputs,progressive=self.progressive,encoded=self.encoded,stack=self.stack),
                        lane=self.lane,key=('lut_bake',self.signature,self.stack)))
                except Exception:
                    self._failed()
                    raise
            return self

        def done(self):
            if self.handle is not None:
                finished=self.handle.retired() if self.handle.cancelled() else self.handle.done()
                if finished:self._retire()
                return finished
            return self.cube is not None or self.protocol.state.phase == 'closed'

        def collect(self,wait=False):
            if self.handle is None:return
            if self.handle.cancelled():
                # Consumer cancellation makes Future.done() true before the
                # physical writer retires. Timer collection must never wait.
                self._retire()
                if wait:self.result()
                return
            payload=self.handle.preview
            if payload is not None and payload is not self._preview_payload:
                if not self.protocol.state.closing:
                    try:self._accept_preview(payload)
                    except Exception:
                        self._failed()
                        raise
            if wait or self.handle.done():self.result()

        def result(self,timeout=None):
            if self.cube is not None:return self.cube
            self.start()
            if self.handle.cancelled():
                deadline=None if timeout is None else time.monotonic()+timeout
                while not self.handle.retired():
                    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('LUT retirement expired')
                    time.sleep(.001)
            try:
                payload=self.handle.result(timeout=timeout)
                # A completed/shared handle may refuse Future.cancel() while
                # this Baker has already lost its logical publication authority.
                if self.protocol.state.closing:
                    self._retire()
                    raise CancelledError('Obsolete LUT result')
                if self.handle.preview is not None and self.preview is None:
                    self._accept_preview(self.handle.preview)
                cube=thaw_cube(payload)
                self._accept_candidate(cube)
            except TimeoutError:
                raise
            except CancelledError:
                self._retire()
                raise
            except Exception:
                self._failed()
                raise
            self.cube=cube;self.device=cube.diagnostics.get('device','')
            self._backend=cube.diagnostics.get('backend',self.backend)
            return cube

        def finish(self):return self.result()

        def finish_preview(self):
            if not self.progressive:raise ValueError('Preview requested for a final-only bake')
            self.start()
            while self.preview is None:
                get_runtime().poll();self.collect()
                if self.preview is not None:break
                if self.cube is not None:
                    payload=freeze_cube(self.cube);payload.pop('prepared_pixels',None)
                    job=get_runtime().submit(JobSpec('pack_lut',dict(cube=payload,encoded=self.encoded,preview=True,
                        preview_signature=lut.signature(self.model,lut.PREVIEW_SIZE)),
                        key=('preview_lut',self.signature,self.encoded)))
                    self._accept_preview(job.result())
                    break
                time.sleep(.001)
            return self.preview

        def advance(self):
            self.start();get_runtime().poll();self.collect()
            return self.cube is None

        def cancel(self):
            effect=self._event('close')
            cancelled=self.handle.cancel() if self.handle is not None and effect.action=='cancel' else effect.accepted
            self._retire()
            return cancelled



    def prepared_pixels(cube,encoded,wait=False):
        """Repack a shared final result on the bulk helper if its atlas differs."""
        expected=(lut.atlas_height(cube.size,4 if encoded else 1),lut.ATLAS_WIDTH,4)
        pixels=getattr(cube,'prepared_pixels',None)
        if pixels is not None and pixels.shape==expected:return pixels
        jobs=getattr(cube,'_packing_jobs',None)
        if jobs is None:jobs={};cube._packing_jobs=jobs
        job=jobs.get(encoded)
        if job is None:
            payload=freeze_cube(cube);payload.pop('prepared_pixels',None)
            job=get_runtime().submit(JobSpec('pack_lut',dict(cube=payload,encoded=encoded),
                key=('pack_lut',cube.signature,encoded)))
            jobs[encoded]=job
        if not wait and not job.done():return None
        pixels=job.result()
        if pixels.shape!=expected:raise ValueError('Helper returned an incompatible LUT atlas')
        cube.prepared_pixels=pixels
        return pixels

    def _set_export_RemoteBaker(value):
        nonlocal RemoteBaker
        RemoteBaker = value

    def _set_export_prepared_pixels(value):
        nonlocal prepared_pixels
        prepared_pixels = value

    return Component(readers={
        "__name__": lambda: __name__,
        'CancelledError': lambda: CancelledError,
        'JobSpec': lambda: JobSpec,
        'RemoteBaker': lambda: RemoteBaker,
        'freeze_cube': lambda: freeze_cube,
        'freeze_model': lambda: freeze_model,
        'get_runtime': lambda: get_runtime,
        'lut': lambda: lut,
        'prepared_pixels': lambda: prepared_pixels,
        'thaw_cube': lambda: thaw_cube,
        'time': lambda: time,
    }, writers={
    'RemoteBaker': _set_export_RemoteBaker, 'prepared_pixels': _set_export_prepared_pixels})
