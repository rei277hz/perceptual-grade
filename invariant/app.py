import asyncio
from copy import deepcopy
import itertools
import queue
import threading


class ClosedError(RuntimeError):
    pass


class BackpressureError(RuntimeError):
    pass


class App:
    def __init__(self, service_factory, *, capacity=256):
        self._commands = queue.Queue(capacity)
        self._events = queue.Queue()
        self._latest = {}
        self._lock = threading.Lock()
        self._tickets = threading.BoundedSemaphore(capacity)
        self._ids = itertools.count(1)
        self._closing = threading.Event()
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(service_factory,),
                                        name='invariant-coordinator', daemon=True)
        self._thread.start()

    def post(self, command):
        if self._closing.is_set() or self.closed.is_set():
            raise ClosedError('application is closing')
        payload = deepcopy(dict(command))
        if not self._tickets.acquire(blocking=False):
            raise BackpressureError('poll notifications before submitting more commands')
        identity = next(self._ids)
        try:
            self._commands.put_nowait((identity, payload))
        except queue.Full as error:
            self._tickets.release()
            raise BackpressureError('command queue is full') from error
        return identity

    def _emit(self, event):
        value = deepcopy(dict(event))
        if value.get('kind') == 'document_closed':
            with self._lock:
                for kind in ('snapshot', 'progress'):
                    self._latest.pop((kind, value.get('document')), None)
        if value.get('kind') in ('progress', 'snapshot'):
            key = (value['kind'], value.get('document', value.get('job', 0)))
            with self._lock:
                self._latest[key] = value
        else:
            self._events.put(value)

    def poll(self, max_events=64):
        if max_events < 1:
            return ()
        result = []
        while len(result) < max_events:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event.get('kind') == 'ack':
                self._tickets.release()
            result.append(event)
        with self._lock:
            while self._latest and len(result) < max_events:
                _, event = self._latest.popitem()
                result.append(event)
        return tuple(result)

    def close_async(self):
        self._closing.set()
        return self.closed

    def _run(self, service_factory):
        try:
            asyncio.run(self._serve(service_factory))
        except BaseException as error:
            self._emit({'kind': 'fatal', 'error': f'{type(error).__name__}: {error}'})
        finally:
            self._closing.set()
            self.closed.set()
            self._emit({'kind': 'closed'})

    async def _serve(self, service_factory):
        service = service_factory(self._emit)
        try:
            await service.start()
            while not self._closing.is_set():
                for _ in range(32):
                    try:
                        identity, command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        await service.handle(command)
                        self._emit({'kind': 'ack', 'request': identity})
                    except Exception as error:
                        self._emit({'kind': 'ack', 'request': identity,
                                    'error': f'{type(error).__name__}: {error}'})
                await service.tick()
                await asyncio.sleep(0.01)
        finally:
            await service.close()
