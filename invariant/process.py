import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
import uuid

from . import lifecycle
from .machine import Machine, Store
from . import _lifecycle_contracts


@dataclass
class Slot:
    index: int
    store: Store
    process: object = None
    reader: object = None
    stderr_reader: object = None
    session: str = ''
    work: dict | None = None
    started: float = 0.0
    cancelling: float = 0.0
    cancel_reason: str = ''
    terminal_seen: bool = False
    stderr: str = ''


class ProcessPool:
    def __init__(self, executable, callback, *, workers=2, memory_limit=512 * 1024 * 1024,
                 timeout=30.0, cancel_grace=1.0):
        if (type(workers) is not int or workers < 1 or type(memory_limit) is not int or memory_limit < workers
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (timeout, cancel_grace))):
            raise ValueError('invalid worker count, memory budget, or deadline')
        self.executable = str(Path(executable).resolve())
        self.callback = callback
        self.memory_limit = memory_limit // workers
        self.timeout = timeout
        self.cancel_grace = cancel_grace
        self.pending = deque()
        self.events = asyncio.Queue()
        self.progress = {}
        machine = Machine(lifecycle, contracts=_lifecycle_contracts)
        self.slots = [Slot(i, Store(machine)) for i in range(workers)]
        self.closing = False
        self.stale_receipts = 0

    async def start(self):
        await asyncio.gather(*(self._spawn(slot) for slot in self.slots))

    async def _spawn(self, slot):
        slot.session = uuid.uuid4().hex
        slot.stderr = ''
        slot.process = await asyncio.create_subprocess_exec(
            self.executable, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024)
        slot.stderr_reader = asyncio.create_task(self._stderr(slot, slot.process))
        try:
            line = await asyncio.wait_for(slot.process.stdout.readline(), 5.0)
            ready = json.loads(line)
            if ready.get('kind') != 'ready' or ready.get('protocol') != 1:
                raise ValueError('worker protocol mismatch')
        except BaseException:
            if slot.process.returncode is None:
                slot.process.kill()
            await slot.process.wait()
            await slot.stderr_reader
            raise RuntimeError(f'worker startup failed: {slot.stderr}')
        slot.reader = asyncio.create_task(self._read(slot.index, slot.session, slot.process))

    async def _stderr(self, slot, process):
        while data := await process.stderr.read(4096):
            slot.stderr = (slot.stderr + data.decode(errors='replace'))[-16384:]

    async def _read(self, index, session, process):
        try:
            while line := await process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('worker message must be an object')
                if message.get('kind') == 'progress':
                    self.progress[index] = (session, message)
                else:
                    await self.events.put((index, session, message))
        except (ValueError, asyncio.LimitOverrunError) as error:
            await self.events.put((index, session, {'kind': 'protocol_error', 'error': str(error)}))
            if process.returncode is None:
                process.kill()
        finally:
            code = await process.wait()
            await self.events.put((index, session, {'kind': 'exited', 'code': code}))

    async def _send(self, slot, command):
        if slot.process is None or slot.process.returncode is not None:
            return False
        try:
            slot.process.stdin.write(json.dumps(command, allow_nan=False).encode() + b'\n')
            await asyncio.wait_for(slot.process.stdin.drain(), 1.0)
            return True
        except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
            if slot.process.returncode is None:
                slot.process.kill()
            return False

    def submit(self, work):
        if self.closing:
            raise RuntimeError('pool is closing')
        if len(self.pending) >= 256:
            raise RuntimeError('worker queue is full')
        self.pending.append(dict(work))

    async def cancel(self, job, *, reason='user'):
        self.pending = deque(item for item in self.pending if item['job'] != job)
        for slot in self.slots:
            if slot.work is not None and slot.work['job'] == job and not slot.cancelling:
                effects = slot.store.dispatch(lifecycle.Event('cancel', slot.store.state.generation))
                if effects.cancel:
                    slot.cancelling = time.monotonic()
                    slot.cancel_reason = reason
                    await self._send(slot, self._identity(slot) | {'kind': 'cancel'})

    def _identity(self, slot):
        return {key: slot.work[key] for key in ('session', 'job', 'revision', 'attempt')}

    async def tick(self):
        for _ in range(128):
            try:
                index, session, message = self.events.get_nowait()
            except asyncio.QueueEmpty:
                break
            slot = self.slots[index]
            if slot.session != session:
                self.stale_receipts += 1
                continue
            await self._message(slot, message)
        for index in tuple(self.progress):
            session, message = self.progress.pop(index)
            slot = self.slots[index]
            if slot.session != session:
                self.stale_receipts += 1
            else:
                await self._message(slot, message)
        now = time.monotonic()
        for slot in self.slots:
            if slot.work is not None:
                if not slot.cancelling and now - slot.started > self.timeout:
                    await self.cancel(slot.work['job'], reason='timeout')
                elif slot.cancelling and now - slot.cancelling > self.cancel_grace:
                    if slot.process.returncode is None:
                        slot.process.kill()
            elif not self.closing and self.pending and slot.process is not None:
                if slot.process.returncode is None and not slot.store.state.closed:
                    work = self.pending.popleft()
                    effects = slot.store.dispatch(lifecycle.Event('start', slot.store.state.generation))
                    if not effects.start:
                        raise RuntimeError('pool attempted admission without a free execution lease')
                    slot.work = work | {'session': slot.session, 'kind': 'run',
                                        'memory_limit': self.memory_limit}
                    slot.started = now
                    slot.cancelling = 0.0
                    slot.cancel_reason = ''
                    slot.terminal_seen = False
                    await self.callback(slot.work, {'kind': 'started', 'worker': slot.index})
                    command = dict(slot.work)
                    if command.get('fault', '').endswith('_once'):
                        command['fault'] = command['fault'][:-5] if command['attempt'] == 0 else ''
                    await self._send(slot, command)

    async def _message(self, slot, message):
        kind = message.get('kind')
        if kind == 'exited':
            work = slot.work
            slot.store.dispatch(lifecycle.Event('died', slot.store.state.generation))
            slot.work = None
            await slot.stderr_reader
            if work is not None:
                await self.callback(work, {'kind': 'crashed', 'code': message['code'],
                                          'reason': slot.cancel_reason,
                                          'error': slot.stderr[-2000:]})
            if not self.closing:
                await self._spawn(slot)
            return
        if kind == 'protocol_error':
            return
        if slot.work is None or any(message.get(key) != value
                                    for key, value in self._identity(slot).items()):
            self.stale_receipts += 1
            return
        if kind in ('complete', 'failed', 'cancelled'):
            if slot.terminal_seen:
                self.stale_receipts += 1
                return
            slot.terminal_seen = True
            slot.store.dispatch(lifecycle.Event('terminal', slot.store.state.generation))
            await self.callback(slot.work, message | {'reason': slot.cancel_reason})
        elif kind == 'quiescent':
            diagnostics = message.get('diagnostics', {})
            valid = isinstance(diagnostics, dict) and all(
                type(diagnostics.get(key)) is int and diagnostics[key] == 0 for key in ('tasks', 'leases'))
            if not slot.terminal_seen or not valid:
                if slot.process.returncode is None:
                    slot.process.kill()
                return
            effects = slot.store.dispatch(lifecycle.Event('quiescent', slot.store.state.generation))
            if effects.reclaim:
                work = slot.work
                slot.work = None
                await self.callback(work, message)
        elif kind == 'progress':
            await self.callback(slot.work, message)

    async def close(self):
        self.closing = True
        self.pending.clear()
        for slot in self.slots:
            if slot.store.state.closed:
                continue
            slot.store.dispatch(lifecycle.Event('close', slot.store.state.generation))
            if slot.work is not None:
                slot.cancel_reason = 'shutdown'
                slot.cancelling = time.monotonic()
            await self._send(slot, {'kind': 'shutdown'})
        deadline = time.monotonic() + self.cancel_grace
        while any(slot.process is not None and slot.process.returncode is None for slot in self.slots):
            await self.tick()
            if time.monotonic() >= deadline:
                for slot in self.slots:
                    if slot.process is not None and slot.process.returncode is None:
                        slot.process.kill()
                break
            await asyncio.sleep(0.01)
        for slot in self.slots:
            if slot.reader is not None:
                await slot.reader
            if slot.stderr_reader is not None:
                await slot.stderr_reader
        while not self.events.empty() or self.progress:
            await self.tick()

    def snapshot(self):
        return {'queued': len(self.pending), 'busy': sum(slot.work is not None for slot in self.slots),
                'execution_leases': sum(slot.store.state.leases for slot in self.slots),
                'stale_receipts': self.stale_receipts,
                'workers': [{'generation': slot.store.state.generation,
                             'busy': slot.store.state.busy, 'leases': slot.store.state.leases}
                            for slot in self.slots]}
