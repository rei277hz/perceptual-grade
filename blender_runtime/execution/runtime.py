"""One foreground-owned service for reusable CPU/GPU helper processes."""
import atexit
from collections import OrderedDict, deque
from collections.abc import Mapping
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, field, replace
from itertools import count
from pathlib import Path
import time
import weakref
import logging
from .diagnostics import log_exception, text
import json

import numpy as np

from . import constants as policy
from ..app import ForeignOwnerError
from ..protocol import Protocol
from .machines import ExecutionMachines
from .contracts import CapacityError, ExecutionDiagnostic, ExecutionError, JobEvent, JobFailure, JobSpec, PROTOCOL, RuntimeLimits, RuntimeProfile, ShutdownReport
from .journal import Session
from .payloads import PayloadStore
from .process import Helper, fingerprint
from .values import seal as seal_values


def _size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_size(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_size(v, seen) for v in value)
    return 0


@dataclass
class _Job:
    spec: JobSpec
    order: int
    attempt: object
    role: str
    retry: object
    workflow: object
    receipt: object = None
    consumers: set = field(default_factory=set)
    stage: object = None
    input_files: list = field(default_factory=list)
    progress: float = 0.
    preview: object = None
    result: object = None
    error: object = None
    finished: bool = False
    fallback_reason: object = None
    diagnostic: object = None


    @property
    def state(self):
        return self.attempt.state

    @property
    def retries(self):
        return self.retry.state.retries


def _forget_consumer(runtime_ref, job_ref, identity):
    runtime, job = runtime_ref(), job_ref()
    if runtime is None:
        return
    if job is not None:
        runtime._forget(identity, job)
    else:
        runtime.events = deque(event for event in runtime.events if event.handle != identity)


class JobHandle:
    """Consumer interest and foreground Future-compatible result access."""
    def __init__(self, runtime, identity, job):
        self.runtime, self.identity, self._job = runtime, identity, job
        self._cancelled = False
        self._finalizer = weakref.finalize(self, runtime._forget, identity, job)

    @property
    def progress(self):
        return self._job.progress

    @property
    def preview(self):
        return self._job.preview

    @property
    def backend(self):
        return self._job.role

    @property
    def generation(self):
        return self._job.state.generation

    @property
    def fallback_reason(self):
        return self._job.fallback_reason

    @property
    def status(self):
        if self._cancelled:
            return 'cancelled'
        if self._job.finished:
            return 'failed' if self._job.error else 'complete'
        return ('queued', 'running', 'cancelling', 'retired')[self._job.state.phase]

    def done(self):
        self.runtime._pump()
        return self._cancelled or self._job.finished

    def running(self):
        return self._job.attempt._query_cleanup("occupied") and not self._cancelled

    def retired(self):
        self.runtime._pump()
        return self._job.finished

    def cancelled(self):
        return self._cancelled

    def cancel(self):
        return self.runtime.cancel(self)

    def result(self, timeout=None):
        self.runtime.wait_exact([self], deadline=None if timeout is None else time.monotonic() + timeout)
        if self._cancelled:
            raise CancelledError('Consumer cancelled')
        if self._job.error is not None:
            raise self._job.error.exception()
        return self._job.result

    def exception(self, timeout=None):
        try:
            self.result(timeout)
        except Exception as error:
            return type(error)(*error.args)
        return None

class ExecutionRuntime:
    def __init__(self, directory, *, app, machines, package, limits=None, launcher=None, profile=None,
                 seal=seal_values, worker_host=None, host_settings=None, environment=None):
        app.require_open()
        if type(machines) is not ExecutionMachines:
            raise TypeError('Execution requires typed machine descriptors')
        self.app, self.machines, self.app_identity = app, machines, app.identity
        self._query_store = app.store(machine=machines.attempt, initial=dict(identity=1, generation=0, interests=1))
        self.profile = profile or RuntimeProfile()
        if type(self.profile) is not RuntimeProfile:
            raise TypeError('Execution requires an immutable RuntimeProfile')
        if not callable(seal):
            raise TypeError('Execution requires a value-sealing function')
        if worker_host is not None and (type(worker_host) is not str or
                not worker_host or not all(part.isidentifier() for part in worker_host.split('.'))):
            raise ValueError('Worker host must be a namespace-relative module name')
        self.seal = seal
        self.worker_host = worker_host
        self.host_settings = json.loads(json.dumps(host_settings or {}))
        self.environment = None if environment is None else dict(environment)
        self.gpu_roles = tuple(dict.fromkeys(role for _, role in self.profile.gpu_probes))
        self.gpu_probe_operations = dict(self.profile.gpu_probes)
        self.limits = limits or RuntimeLimits()
        self.session = Session(directory)
        self.payloads = PayloadStore(self.session.path, self.limits.payload_bytes)
        self.package = Path(package).absolute()
        self.fingerprint = fingerprint(self.package)
        self.launcher = launcher
        self.helpers = {}
        self.generations = dict.fromkeys(self.profile.lanes, 0)
        self._startup = {lane: app.store(machine=machines.startup, initial=dict(failures=0))
                         for lane in self.profile.lanes if lane not in self.gpu_roles}
        self.jobs = {}
        self.handles = weakref.WeakValueDictionary()
        self.cache = OrderedDict()
        self._serial = count(1)
        self.events = deque()
        self.gpu_failures = {role: None for role in self.gpu_roles}
        self._gpu = {role: app.store(machine=machines.gpu) for role in self.gpu_roles}
        self.gpu_probes = {role: None for role in self.gpu_roles}
        self.gpu_diagnostics = {role: deque(maxlen=8) for role in self.gpu_roles}
        self.diagnostics = deque(maxlen=8)
        self._starting_probe = None
        self.fenced = self.closed = self._pumping = False
        self._shutdown_report = None
        self.cleanup_blocked = False
        self._exit_callback = self.shutdown
        atexit.register(self._exit_callback)

    @property
    def gpu_healths(self):
        return {role: store.state for role, store in self._gpu.items()}

    @property
    def startup_failures(self):
        return {role: store.state.failures for role, store in self._startup.items()}

    def _attempt_event(self, job, kind, generation=None):
        event = self.machines.event(self.machines.attempt, kind=kind, identity=job.state.identity,
                                    generation=job.state.generation if generation is None else generation)
        dispatch = job.attempt._dispatch_cleanup if kind in ('detach', 'retire') else job.attempt.dispatch
        return dispatch(event).accepted

    def _gpu_event(self, role, kind, attempt, generation, validated=False):
        event = self.machines.event(self.machines.gpu, kind=kind, attempt=attempt,
                                   generation=generation, validated=validated)
        return self._gpu[role]._dispatch_cleanup(event).accepted

    def _gpu_available(self, role):
        return self._gpu[role]._query_cleanup('gpu_available')

    def _new_attempt(self, identity, generation, interests=1):
        return self.app.store(machine=self.machines.attempt,
                              initial=dict(identity=identity, generation=generation, interests=interests))

    def _new_job(self, spec, identity, generation, role):
        workflow = Protocol(self.app, self.machines.coordinator)
        # Coordinator epochs are logical workflow epochs. The transport keeps
        # the independently validated session/lane/helper/attempt identity.
        workflow.require('submit', 'start', job=identity, generation=1, attempt=1,
                         gpu=role in self.gpu_roles,
                         retryable=spec.operation not in self.gpu_probe_operations)
        return _Job(spec, identity, self._new_attempt(identity, generation), role,
                    self.app.store(machine=self.machines.retry, initial=dict(retries=0)), workflow)

    def _workflow(self, job, kind, *, receipt=False, **values):
        if receipt:
            if job.receipt is None:
                raise ExecutionError('Native receipt has no admitted workflow identity')
            physical, logical = job.receipt
            if physical != (job.role, job.state.generation, job.state.identity):
                raise ExecutionError('Native receipt belongs to a superseded attempt')
            values.update(zip(('job', 'generation', 'attempt'), logical))
        return job.workflow.event(kind, **values)

    def _require_retired(self, job):
        if any(helper.active is job for helper in self.helpers.values()):
            raise ExecutionError('Execution publication requires confirmed physical retirement')

    def _require_handle(self, handle):
        if (type(handle) is not JobHandle or handle.runtime is not self or
                self.handles.get(handle.identity) is not handle):
            raise ForeignOwnerError('Execution handle belongs to another runtime')

    # Convenience accessors for profiles with the conventional 'gpu' lane.
    # Multi-device profiles can query each named lane through gpu_status().
    @property
    def gpu_failure(self):
        return self.gpu_failures['gpu']

    @gpu_failure.setter
    def gpu_failure(self, value):
        self.gpu_failures['gpu'] = value

    @property
    def gpu_health(self):
        return self.gpu_healths['gpu']

    @property
    def gpu_probe(self):
        return self.gpu_probes['gpu']

    @gpu_probe.setter
    def gpu_probe(self, value):
        self.gpu_probes['gpu'] = value

    def submit(self, spec):
        if self.fenced:
            raise ExecutionError('Execution runtime is shutting down')
        if (type(spec) is not JobSpec or type(spec.operation) is not str or
                type(spec.lane) is not str or type(spec.priority) is not int):
            raise TypeError('Execution requires a value-only JobSpec')
        # Seal before the job, its staging generator, or a finalizer can retain
        # the caller's graph. The public dataclass is not an ownership boundary.
        spec = replace(spec, inputs=self.seal(spec.inputs), key=self.seal(spec.key))
        if spec.lane not in self.generations:
            raise ValueError('Unknown execution lane: ' + spec.lane)
        probe_role = self.gpu_probe_operations.get(spec.operation)
        probe = probe_role is not None
        if probe and (self._starting_probe != probe_role or spec.lane != probe_role or spec.key is not None):
            raise ExecutionError('GPU probes require explicit retry and cannot use a cache')
        if len(self.handles) >= self.limits.pending_jobs * 4:
            raise CapacityError('Execution consumer budget exhausted')
        if spec.operation in self.profile.uncached_operations and spec.key is not None:
            raise ExecutionError('Operation requires a fresh helper acknowledgement')
        key = None if spec.key is None else (spec.lane, spec.operation, repr(spec.key))
        job = self.cache.get(key) if key is not None else None
        if job is None:
            job = next((item for item in self.jobs.values()
                        if not item.finished and item.state.phase in (policy.QUEUED, policy.RUNNING)
                        and key is not None and (item.spec.lane, item.spec.operation, repr(item.spec.key)) == key), None)
        if job is not None and not job.finished:
            if not self._attempt_event(job, 'attach'):
                job = None
        if job is None:
            active = sum(not item.finished for item in self.jobs.values())
            total = sum(_size(item.spec.inputs) for item in self.jobs.values() if not item.finished) + _size(spec.inputs)
            if not self._query_store.query('can_admit', jobs=active, size=total, max_jobs=self.limits.pending_jobs, max_size=self.limits.payload_bytes):
                raise CapacityError('Execution queue or payload budget exhausted')
            identity = next(self._serial)
            role = (self.profile.fallback_lane if spec.lane in self.gpu_roles and not probe and
                    not self._gpu_available(spec.lane) else spec.lane)
            job = self._new_job(spec, identity, self.generations[role], role)
            if role != spec.lane:
                job.fallback_reason = self.gpu_failures[spec.lane] or 'GPU validation is in progress'
            self.jobs[identity] = job
        identity = next(self._serial)
        handle = JobHandle(self, identity, job)
        self.handles[identity] = handle
        job.consumers.add(identity)
        if job.finished:
            self._event(JobEvent(identity, 'complete', job.result))
        return handle

    def cancel(self, handle):
        self._require_handle(handle)
        if handle._cancelled or handle._job.finished:
            return False
        handle._cancelled = True
        self._detach(handle.identity, handle._job)
        return True

    def _detach(self, identity, job):
        if job.finished or identity not in job.consumers:
            return
        job.consumers.discard(identity)
        self._attempt_event(job, 'detach')
        self._event(JobEvent(identity, 'cancelled'))
        if job.state.phase == policy.CANCELLING:
            self._workflow(job, 'cancel')
            helper = self.helpers.get(job.role)
            if helper is not None and helper.ready and helper.active is job:
                self._send(helper, dict(kind='cancel', attempt=job.state.identity))
        elif job.state.phase == policy.RETIRED:
            if job.stage is not None:
                job.stage.close()
                job.stage = None
            self._finish(job, error=CancelledError('No interested consumers'))

    def _forget(self, identity, job):
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.app._defer_finalizer(_forget_consumer, weakref.ref(self), weakref.ref(job), identity)
            return
        self._detach(identity, job)
        job.consumers.discard(identity)
        self.events = deque(event for event in self.events if event.handle != identity)

    def _event(self, event):
        # Keep terminal events until consumed. Progress/preview events retain
        # only their latest value per live consumer, bounding retained memory.
        self.events = deque(old for old in self.events if old.handle != event.handle
                            or (event.kind in ('progress', 'preview') and old.kind != event.kind))
        self.events.append(event)

    def _send(self, helper, message):
        message.update(session=self.session.identity, generation=helper.generation)
        try:
            helper.send(message)
            return True
        except (OSError, ValueError, BufferError, ExecutionError) as error:
            helper.failure = 'Helper send failed: ' + text(error)
            helper.kill()
            return False

    def _notify(self, job, kind, value=None):
        for identity in tuple(job.consumers):
            handle = self.handles.get(identity)
            if handle is None:
                continue
            self._event(JobEvent(identity, kind, value))

    def _finish(self, job, result=None, error=None):
        self._require_retired(job)
        state = job.workflow.state
        if state.phase not in ('complete', 'failed', 'cancelled', 'closed'):
            if isinstance(error, CancelledError):
                self._workflow(job, 'cancel')
                decision = self._workflow(job, 'retired', accepted=True)
            elif error is not None and state.phase == 'starting':
                decision = self._workflow(job, 'rejected', accepted=True)
            else:
                self._workflow(job, 'complete' if error is None else 'failed', receipt=job.receipt is not None)
                decision = self._workflow(job, 'retired', accepted=True)
            if error is None and decision.action != 'deliver':
                raise ExecutionError('Coordinator did not authorize result delivery')
        elif error is None and state.phase != 'complete':
            raise ExecutionError('Coordinator has not certified result delivery')
        if (job.spec.operation in self.gpu_probe_operations and error is not None
                and self.gpu_healths[job.spec.lane].phase == policy.GPU_PROBING):
            self._gpu_failed(job, text(error))
        self._attempt_event(job, 'retire')
        failure = None if error is None else JobFailure(text(error), isinstance(error, CancelledError))
        job.finished, job.result, job.error = True, result, failure
        job.progress = 1. if error is None else job.progress
        self.payloads.release_files(job.input_files)
        job.input_files.clear()
        self.jobs.pop(job.order, None)
        job.spec = replace(job.spec, inputs={})
        if error is None and job.spec.key is not None:
            key = (job.spec.lane, job.spec.operation, repr(job.spec.key))
            self.cache[key] = job
            self.cache.move_to_end(key)
            while len(self.cache) > 16 or sum(_size((j.result, j.preview)) for j in self.cache.values()) > self.limits.result_bytes:
                self.cache.popitem(last=False)
        self._notify(job, 'complete' if error is None else 'failed', result if error is None else text(error))

    def _diagnostic(self, job, reason, helper=None, message=None):
        message = message or {}
        details = message.get('diagnostic') or {}
        return ExecutionDiagnostic(job.spec.operation, str(reason),
            str(message.get('traceback', '')), helper.error() if helper else '',
            str(details.get('backend', getattr(helper, 'backend', 'GPU')))[-128:],
            str(details.get('device', ''))[-512:], self.fingerprint, self.session.identity,
            job.state.identity, helper.generation if helper else job.state.generation,
            helper.process.pid if helper else 0, str(helper.log_path) if helper else '',
            str(self.session.path / f'{job.role}-diagnostic-{job.state.generation}-{job.state.identity}.json'),
            role=job.role)

    def _gpu_failed(self, job, reason):
        role = job.role
        accepted = self._gpu_event(role, 'failed',
            job.state.identity, job.state.generation)
        if not accepted:return
        self.gpu_failures[role] = reason
        diagnostic = job.diagnostic or self._diagnostic(job, reason)
        if not self.diagnostics or self.diagnostics[-1] != diagnostic:
            self.diagnostics.append(diagnostic)
            self.gpu_diagnostics[role].append(diagnostic)
            try:
                Path(diagnostic.diagnostic_path).write_text(json.dumps(asdict(diagnostic), indent=2), encoding='utf8')
            except OSError:
                log_exception(logging.getLogger(__name__),'Could not persist GPU diagnostic')
            logging.getLogger(__name__).warning('%s unavailable; using CPU. %s', role, text(diagnostic))

    def _retry(self, job, reason, kind='failed'):
        # A terminal receipt or a confirmed helper exit has already retired
        # physical occupancy before this function creates another attempt.
        self._require_retired(job)
        if self.fenced or not job.consumers:
            self._workflow(job, 'cancel')
        if kind == 'rebind' and job.workflow.state.phase == 'starting':
            decision = self._workflow(job, 'rebind', accepted=True)
        else:
            if job.workflow.state.phase not in ('retiring', 'cancelled', 'closed'):
                self._workflow(job, kind, receipt=job.receipt is not None)
            decision = self._workflow(job, 'retired', accepted=True)
        self._attempt_event(job, 'retire')
        if job.stage is not None:
            job.stage.close()
            job.stage = None
        self.payloads.release_files(job.input_files)
        job.input_files.clear()
        if job.spec.operation in self.gpu_probe_operations:
            self._gpu_failed(job, reason)
        if job.workflow.state.phase in ('cancelled', 'closed'):
            self._finish(job, error=CancelledError(reason))
        elif decision.action == 'fallback':
            if job.diagnostic is not None or self._gpu_available(job.role):
                self._gpu_failed(job, reason)
            job.fallback_reason = reason
            job.role = self.profile.fallback_lane
        elif decision.action == 'ask_retry':
            accepted = job.retry.dispatch(self.machines.event(self.machines.retry, kind=kind)).accepted
            decision = self._workflow(job, 'retry_decision', accepted=accepted)
        if not job.finished and decision.action == 'fail':
            self._finish(job, error=ExecutionError(reason))
        if not job.finished and decision.action not in ('start', 'fallback'):
            raise ExecutionError('Coordinator did not authorize another execution attempt')
        if not job.finished:
            job.attempt = self._new_attempt(next(self._serial), self.generations[job.role], interests=len(job.consumers))
            job.stage, job.preview, job.progress = None, None, 0.
            job.receipt = None

    def _startup_failed(self, role, reason):
        """Retire a failed CPU helper generation without charging job retries."""
        queued = tuple(job for job in self.jobs.values()
                       if job.role == role and job.state.phase == policy.QUEUED)
        if not queued:
            return
        accepted = self._startup[role].dispatch(self.machines.event(self.machines.startup, kind='failed')).accepted
        if not accepted:
            self._startup[role] = self.app.store(machine=self.machines.startup, initial=dict(failures=0))
        for job in queued:
            self._retry(job, reason, kind='rebind' if accepted else 'failed')

    def _message(self, helper, message):
        if message.get('session') != self.session.identity or message.get('generation') != helper.generation:
            return
        if message['kind'] == 'hello':
            if (message.get('protocol') != PROTOCOL or message.get('token') != helper.token
                    or message.get('fingerprint') != self.fingerprint):
                raise ExecutionError('Helper handshake is incompatible')
            helper.ready = True
            helper.backend = message.get('backend', 'GPU')
            helper.last_progress = time.monotonic()
            return
        job = helper.active
        if job is None or message.get('attempt') != job.state.identity:
            return
        helper.last_progress = time.monotonic()
        kind = message['kind']
        if kind == 'progress':
            job.progress = float(message['value'])
            self._notify(job, 'progress', job.progress)
        elif kind == 'preview':
            if job.state.phase == policy.RUNNING:
                job.preview = self.seal(self.payloads.decode(message['value']))
                self._notify(job, 'preview', job.preview)
            else:
                self.payloads.release_files(message.get('files', []))
        elif kind in ('complete', 'failed', 'cancelled'):
            # Keep the physical owner attached until its payload validates. A
            # malformed receipt must retire through confirmed process exit.
            result = None
            if kind == 'complete' and job.state.phase != policy.CANCELLING:
                result = self.seal(self.payloads.decode(message['value']))
                if job.spec.operation in self.gpu_probe_operations:
                    role = self.gpu_probe_operations[job.spec.operation]
                    valid = (helper.role == role and isinstance(result, Mapping)
                             and result.get('conformance') is True and result.get('backend') == 'GPU')
                    accepted = self._gpu_event(role, 'validated',
                        job.state.identity, helper.generation, valid)
                    if not accepted:raise ExecutionError('GPU probe did not validate the current helper')
                    self.gpu_failures[role] = None
            helper.active = None
            if job.state.phase == policy.CANCELLING or kind == 'cancelled':
                self.payloads.release_files(message.get('files', []))
                self._finish(job, error=CancelledError('Computation cancelled'))
            elif kind == 'failed':
                job.diagnostic = self._diagnostic(job, message.get('error', 'Helper failed'), helper, message)
                self._retry(job, str(message.get('error', 'Helper failed')))
            else:
                self._finish(job, result=result)
        else:
            raise ExecutionError('Unknown helper response')

    def _dead(self, role, helper, reason):
        # Call only after OS exit confirmation; cancellation is not an exit.
        if helper.process.poll() is None:
            return
        # A killed producer can leave an incomplete or unsent result. Files
        # already adopted by a foreground view retain that view's lifetime.
        abandoned = [path.name for path in self.session.path.glob(f'array-{role}-{helper.generation}-*.npy')
                     if path not in self.payloads.owned]
        self.payloads.release_files(abandoned)
        job, helper.active = helper.active, None
        if job is not None:job.diagnostic = self._diagnostic(job, reason, helper)
        if (job is None and role in self.gpu_roles and not self.fenced
                and not getattr(helper, 'retiring_for_probe', False)
                and self._gpu_available(role)):
            identity = next(self._serial)
            failure = self._new_job(JobSpec(role + '_helper', {}, lane=role), identity, helper.generation, role)
            failure.diagnostic = self._diagnostic(failure, reason, helper)
            self._gpu_failed(failure, reason)
        helper.close()
        self.helpers.pop(role, None)
        if job is not None:
            self._retry(job, reason, kind='crashed')
        if getattr(helper, 'retiring_for_probe', False):return
        if job is None and role not in self.gpu_roles:
            self._startup_failed(role, reason)
            return
        for queued in tuple(self.jobs.values()):
            if queued is not job and queued.role == role and queued.state.phase == policy.QUEUED:
                queued.attempt = self._new_attempt(queued.state.identity, helper.generation, queued.state.interests)
                queued.diagnostic = self._diagnostic(queued, reason, helper)
                self._retry(queued, reason, kind='rebind')

    def _pump(self, budget_ms=4):
        self.app.drain_finalizers()
        if self.closed or self._pumping:
            return
        self._pumping = True
        end = time.monotonic() + budget_ms / 1000
        try:
            for role, helper in tuple(self.helpers.items()):
                try:
                    for message in helper.poll():
                        self._message(helper, message)
                except (OSError, EOFError, ValueError, KeyError, TypeError, ExecutionError) as error:
                    helper.kill()
                    helper.failure = getattr(helper, 'failure', text(error))
                if helper.process.poll() is not None:
                    self._dead(role, helper, getattr(helper, 'failure', 'Helper exited: ' + helper.error()))
                    continue
                now = time.monotonic()
                if ((not helper.ready and now - helper.started > self.limits.startup_seconds)
                        or (helper.active and now - helper.last_progress > self.limits.stalled_seconds)):
                    helper.failure = 'Helper startup or progress deadline exceeded'
                    helper.kill()
            if not self.fenced:
                for role in self.gpu_roles:
                    if self._gpu_available(role):continue
                    for queued in tuple(self.jobs.values()):
                        if (queued.role == role and queued.spec.operation not in self.gpu_probe_operations
                                and queued.state.phase == policy.QUEUED):
                            self._retry(queued, self.gpu_failures[role] or 'GPU validation is in progress')
                for role in self.generations:
                    pending = sorted((j for j in self.jobs.values() if j.role == role and j.state.phase == policy.QUEUED),
                                     key=lambda j: (-j.spec.priority, j.order))
                    if not pending:
                        continue
                    helper = self.helpers.get(role)
                    if helper is None:
                        self.generations[role] += 1
                        pending[0].attempt = self._new_attempt(pending[0].state.identity, self.generations[role], pending[0].state.interests)
                        try:
                            helper = Helper(self.session, role, self.generations[role], self.package,
                                            self.fingerprint, self.launcher, profile=self.profile,
                                            worker_host=self.worker_host, host_settings=self.host_settings,
                                            environment=self.environment)
                            self.helpers[role] = helper
                        except Exception as error:
                            reason = 'Helper launch failed: ' + text(error)
                            if role in self.gpu_roles:
                                self._retry(pending[0], reason)
                            else:
                                self._startup_failed(role, reason)
                            continue
                    if not helper.ready or helper.active is not None or getattr(helper, 'retiring_for_probe', False):
                        continue
                    job = pending[0]
                    if job.stage is None:
                        job.attempt = self._new_attempt(job.state.identity, helper.generation, interests=job.state.interests)
                        job.stage = self.payloads.encode_iter(job.spec.inputs, job.input_files)
                    try:
                        while time.monotonic() < end:
                            next(job.stage)
                    except StopIteration as staged:
                        job.stage = None
                        accepted = self._attempt_event(job, 'start', helper.generation)
                        if not accepted:
                            raise ExecutionError('Attempt admission rejected')
                        if not self._workflow(job, 'started').accepted:
                            raise ExecutionError('Coordinator rejected native worker admission')
                        state = job.workflow.state
                        job.receipt = ((role, helper.generation, job.state.identity),
                                       (state.job, state.generation, state.attempt))
                        helper.active = job
                        if role not in self.gpu_roles:
                            self._startup[role].dispatch(self.machines.event(self.machines.startup, kind='dispatch'))
                        helper.last_progress = time.monotonic()
                        self._send(helper, dict(kind='run', attempt=job.state.identity,
                                                operation=job.spec.operation, inputs=staged.value))
                    except Exception as error:
                        job.stage = None
                        if helper.active is job:
                            helper.failure = 'Dispatch failed: ' + text(error)
                            helper.kill()
                        else:
                            self._finish(job, error=ExecutionError('Payload staging failed: ' + text(error)))
                    if time.monotonic() >= end:
                        break
            self.payloads.collect()
        finally:
            self._pumping = False

    def poll(self, budget_ms=4):
        self._pump(budget_ms)
        events = list(self.events)
        self.events.clear()
        return events

    def wait_exact(self, handles, deadline=None):
        handles = tuple(handles)
        for handle in handles:
            self._require_handle(handle)
        while any(not h._cancelled and not h._job.finished for h in handles):
            self._pump(10)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Exact computation wait expired')
            time.sleep(.001)
        results = []
        for handle in handles:
            if handle._cancelled:
                raise CancelledError('Consumer cancelled')
            if handle._job.error:
                raise handle._job.error.exception()
            results.append(handle._job.result)
        return results

    def retry_gpu(self, role='gpu'):
        """Queue one uncached GPU-only probe; outputs and CPU cache stay valid."""
        if role not in self.gpu_roles:raise ValueError('Unknown GPU role: ' + str(role))
        if self.fenced:raise ExecutionError('Execution runtime is shutting down')
        probe = self.gpu_probes[role]
        if probe is not None and not probe.done():return probe
        helper = self.helpers.get(role)
        if helper is not None and helper.active is not None:
            raise ExecutionError('A GPU job is still retiring; retry after it stops')
        self._starting_probe = role
        operation = next(operation for operation, probe_role in self.gpu_probe_operations.items() if probe_role == role)
        try:handle = self.submit(JobSpec(operation, {}, lane=role))
        finally:self._starting_probe = None
        generation = self.generations[role] + 1
        handle._job.attempt = self._new_attempt(handle._job.state.identity, generation, handle._job.state.interests)
        accepted = self._gpu_event(role, 'probe', handle._job.state.identity, generation)
        if not accepted:
            handle.cancel()
            raise ExecutionError('GPU probe admission rejected')
        self.gpu_probes[role] = handle
        if helper is not None:
            helper.retiring_for_probe = True
            helper.kill()
        return handle

    def gpu_status(self, role='gpu'):
        if role not in self.gpu_roles:raise ValueError('Unknown GPU role: ' + str(role))
        diagnostics = self.gpu_diagnostics[role]
        return {'state': ('ready', 'fallback', 'probing')[self.gpu_healths[role].phase],
                'reason': self.gpu_failures[role] or '',
                'diagnostic': diagnostics[-1] if diagnostics else None}

    def physical_terminal(self):
        """Physical retirement is separate from the historical journal verdict."""
        return bool(self.closed and self.fenced and self.session.lease.file.closed
                    and self._query_store._query_cleanup('can_clean', fenced=self.fenced,
                        attempts=len(self.jobs), helpers=len(self.helpers), payloads=len(self.payloads.owned)))

    def shutdown(self, deadline=None):
        if self.physical_terminal():
            return self._shutdown_report
        previous = self._shutdown_report
        # A retry may reap a previously unkillable helper or release a locked
        # payload. Admission stays fenced; prior dirty evidence stays dirty.
        self.closed = False
        self.fenced = True
        end = time.monotonic() + self.limits.shutdown_seconds if deadline is None else deadline
        errors = [] if previous is None else list(previous.errors)
        for handle in tuple(self.handles.values()):
            self.cancel(handle)
        for helper in self.helpers.values():
            if helper.ready:
                self._send(helper, {'kind': 'shutdown'})
            else:
                helper.terminate()
        while self.helpers and time.monotonic() < end:
            self._pump(10)
            time.sleep(.005)
        for role, helper in tuple(self.helpers.items()):
            try:
                helper.kill()
                helper.process.wait(timeout=2)
                self._dead(role, helper, 'Runtime shutdown')
            except Exception as error:
                errors.append(text(error))
        for job in tuple(self.jobs.values()):
            if not job.attempt._query_cleanup('occupied'):
                self._finish(job, error=CancelledError('Runtime closed'))
        if not self.helpers:
            self.cache.clear()
            self.events.clear()
            try:
                self.payloads.owned.update(self.session.path.glob('array-*.npy'))
                self.payloads.close()
            except Exception as error:
                errors.append(text(error))
        retired = bool(self._query_store._query_cleanup('can_clean', fenced=self.fenced,
            attempts=len(self.jobs), helpers=len(self.helpers), payloads=len(self.payloads.owned)))
        clean = retired and not errors and not self.cleanup_blocked and (previous is None or previous.clean)
        # Keep the lease while physical resources remain; a different App must
        # not recover the directory of a helper whose exit is unconfirmed.
        if retired:
            try:
                self.session.close(clean)
            except Exception as error:
                clean = False
                errors.append(text(error))
        self.closed = True
        self._shutdown_report = ShutdownReport(clean, tuple(errors))
        if self.physical_terminal():
            atexit.unregister(self._exit_callback)
        return self._shutdown_report
