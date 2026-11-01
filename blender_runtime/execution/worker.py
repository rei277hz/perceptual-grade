"""Helper main-thread handlers and a control-only cancellation reader."""
from concurrent.futures import CancelledError
import os
import queue
import socket
import threading
import traceback

from .contracts import PROTOCOL, RuntimeProfile
from .journal import Lease
from .payloads import PayloadStore
from .process import fingerprint
from .transport import frame, receive


class Context:
    def __init__(self, role, job, cancelled, send, payloads, *, generation=0, session=''):
        self.role = self.lane = role
        self.generation = generation
        self.session = session
        self.job = job
        self.cancelled = cancelled
        self.send = send
        self.payloads = payloads
        self.unclaimed = []

    def checkpoint(self):
        if self.cancelled.is_set():
            raise CancelledError('Execution request was cancelled')

    def progress(self, value):
        self.checkpoint()
        self.send(dict(kind='progress', attempt=self.job['attempt'], value=float(value)))

    def preview(self, value):
        self.checkpoint()
        files = self.unclaimed
        encoded = self.payloads.encode(value, files)
        self.send(dict(kind='preview', attempt=self.job['attempt'], value=encoded, files=files))
        self.transferred()

    def transferred(self):
        for name in self.unclaimed:
            self.payloads.owned.discard(self.payloads.path / name)
        self.unclaimed.clear()


from importlib import import_module


def run(config, namespace):
    settings = config.get('profile', {})
    # JSON represents tuples as lists; validate the reconstructed immutable
    # profile before importing any extension-specific operation module.
    settings = dict(settings)
    for name in ('lanes', 'uncached_operations'):
        if name in settings:
            settings[name] = tuple(settings[name])
    if 'gpu_probes' in settings:
        settings['gpu_probes'] = tuple(tuple(pair) for pair in settings['gpu_probes'])
    profile = RuntimeProfile(**settings)
    dispatch = import_module(namespace + '.' + profile.dispatcher).dispatch
    gpu_roles = tuple(role for _, role in profile.gpu_probes)
    host = (None if config.get('worker_host') is None
            else import_module(namespace + '.' + config['worker_host']))
    metadata = {} if host is None else host.initialize(config.get('host_settings', {}))
    lease = Lease(os.path.join(config['directory'], 'lease'))
    sock = socket.create_connection(tuple(config['address']), timeout=30)
    sock.settimeout(None)
    send_lock = threading.Lock()
    def send(message):
        message.update(session=config['session'], generation=config['generation'])
        with send_lock:
            sock.sendall(frame(message))
    send(dict(kind='hello', protocol=PROTOCOL, token=config['token'],
              fingerprint=fingerprint(config['package']), **metadata))
    commands = queue.Queue()
    cancelled = threading.Event()
    stopping = threading.Event()
    active = [None]
    cancelled_attempts = set()
    state_lock = threading.Lock()
    def controls():
        try:
            while True:
                message = receive(sock)
                if message.get('session') != config['session'] or message.get('generation') != config['generation']:
                    raise ValueError('Parent identity mismatch')
                if message['kind'] == 'cancel':
                    with state_lock:
                        cancelled_attempts.add(message['attempt'])
                        if message['attempt'] == active[0]:
                            cancelled.set()
                elif message['kind'] == 'shutdown':
                    stopping.set()
                    cancelled.set()
                    commands.put(None)
                    return
                elif message['kind'] == 'run':
                    commands.put(message)
                else:
                    raise ValueError('Unknown parent command')
        except BaseException:
            # No Blender/GPU cleanup on this thread. The journal handles abrupt
            # exit. Native calls retaining the GIL can delay this safeguard.
            os._exit(70)
    reader = threading.Thread(target=controls, name='Extension helper control', daemon=True)
    reader.start()
    payloads = PayloadStore(config['directory'])
    try:
        while not stopping.is_set():
            job = commands.get()
            if job is None:
                break
            with state_lock:
                active[0] = job['attempt']
                cancelled.clear()
                if job['attempt'] in cancelled_attempts or stopping.is_set():
                    cancelled.set()
            context = Context(config['role'], job, cancelled, send, payloads,
                              generation=config['generation'], session=config['session'])
            payloads.prefix = f"array-{config['role']}-{config['generation']}-{job['attempt']}-"
            try:
                inputs = payloads.decode(job['inputs'], copy=True)
                context.checkpoint()
                result = dispatch(job['operation'], inputs, context)
                context.checkpoint()
                files = context.unclaimed
                value = payloads.encode(result, files)
                context.checkpoint()
                send(dict(kind='complete', attempt=job['attempt'], value=value, files=files))
                context.transferred()
                del inputs, result, value
            except CancelledError:
                send(dict(kind='cancelled', attempt=job['attempt']))
            except Exception as error:
                trace = traceback.format_exc()
                diagnostic = {'backend': '', 'device': ''}
                if host is not None:
                    diagnostic.update(host.diagnostic(context.role in gpu_roles))
                send(dict(kind='failed', attempt=job['attempt'], error=type(error).__name__ + ': ' + str(error),
                          traceback=trace, diagnostic=diagnostic))
            finally:
                payloads.release_files(context.unclaimed)
                with state_lock:
                    cancelled_attempts.discard(job['attempt'])
                    active[0] = None
    finally:
        stopping.set()
        # Normal shutdown is requested by the reader, which is already done.
        reader.join(timeout=2)
        sock.close()
        lease.close()
