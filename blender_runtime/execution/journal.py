"""Durable session records and leases; never infer ownership from a PID."""
import json
import logging
import os
from pathlib import Path
import shutil
import uuid


class Lease:
    def __init__(self, path, *, exclusive=False, create=False):
        self.file = open(path, 'a+b' if create else 'r+b')
        try:
            if os.name == 'nt':
                import ctypes
                import msvcrt
                from ctypes import wintypes
                class Overlapped(ctypes.Structure):
                    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                                ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                                ('hEvent', wintypes.HANDLE)]
                self._overlap = Overlapped()
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                             wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                self._handle = msvcrt.get_osfhandle(self.file.fileno())
                if not kernel.LockFileEx(self._handle, 1 | (2 if exclusive else 0), 0, 1, 0,
                                         ctypes.byref(self._overlap)):
                    raise BlockingIOError(ctypes.get_last_error(), 'Session is leased')
            else:
                import fcntl
                fcntl.flock(self.file, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise

    def close(self):
        self.file.close()


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.writing')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def recover(root):
    recovered = []
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob('session-*'):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            lease = Lease(path / 'lease', exclusive=True)
        except (OSError, ValueError):
            continue
        try:
            try:
                record = json.loads((path / 'journal.json').read_text())
            except (OSError, ValueError):
                record = {'clean': False}
            # Keep the locked file until every other owned resource is gone.
            # Keep the journal too if Windows still has a consumer mapping.
            for child in sorted(path.iterdir(), key=lambda child: child.name == 'journal.json'):
                if child.name == 'lease':
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            recovered.append((path.name, bool(record.get('clean'))))
        except OSError:
            from .diagnostics import Failure
            import sys
            logging.getLogger(__name__).warning('Execution session cleanup deferred: %s\n%s',
                                                str(path), Failure.capture(sys.exception()).detail)
            continue
        finally:
            lease.close()
        try:
            (path / 'lease').unlink()
            path.rmdir()
        except OSError:
            pass
    return tuple(recovered)


class Session:
    def __init__(self, root):
        self.recovered = recover(root)
        self.identity = uuid.uuid4().hex
        self.path = Path(root) / ('session-' + self.identity)
        self.path.mkdir(mode=0o700)
        self.lease = Lease(self.path / 'lease', create=True)
        self.record = {'schema': 1, 'session': self.identity, 'pid': os.getpid(), 'clean': False}
        durable_json(self.path / 'journal.json', self.record)

    def close(self, clean):
        try:
            if clean:
                durable_json(self.path / 'journal.json', dict(self.record, clean=True))
        finally:
            self.lease.close()
