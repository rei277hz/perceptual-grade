"""Explicit primitive/array transport. No pickled Python object graphs."""
from collections import deque
import math
import os
from pathlib import Path
from types import MappingProxyType
import uuid
import weakref
import numpy as np

CHUNK = 256 * 1024


class PayloadStore:
    def __init__(self, directory, maximum=512 * 1024 * 1024):
        self.path = Path(directory)
        self.maximum = maximum
        self.owned = set()
        self.mapped = {}
        self.prefix = 'array-'

    @property
    def bytes(self):
        total = 0
        for path in self.path.glob('array-*.npy'):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass  # Another helper just discarded an unfinished payload.
        return total

    def encode_iter(self, value, files):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.kind not in 'bifu':
                raise ValueError('Only numerical arrays can cross the helper boundary')
            if value.nbytes > self.maximum or self.bytes + value.nbytes > self.maximum:
                raise MemoryError('Execution payload budget exhausted')
            name = self.prefix + uuid.uuid4().hex + '.npy'
            path = self.path / name
            self.owned.add(path)
            files.append(name)
            mapped = np.lib.format.open_memmap(path, mode='w+', dtype=value.dtype, shape=value.shape)
            try:
                source = np.asarray(value)
                # nditer avoids a large contiguous temporary for strided inputs.
                iterator = np.nditer(source, flags=['external_loop', 'buffered', 'zerosize_ok'],
                                     order='C', buffersize=max(1, CHUNK // value.dtype.itemsize))
                output = mapped.reshape(-1)
                offset = 0
                for chunk in iterator:
                    output[offset:offset + chunk.size] = chunk
                    offset += chunk.size
                    yield None
                mapped.flush()
            finally:
                mapped._mmap.close()
            return {'$array': name, 'shape': list(value.shape), 'dtype': value.dtype.str,
                    'bytes': value.nbytes}
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {'$float': repr(value)}
        if isinstance(value, tuple):
            result = []
            for item in value:
                result.append((yield from self.encode_iter(item, files)))
            return {'$tuple': result}
        if isinstance(value, list):
            result = []
            for item in value:
                result.append((yield from self.encode_iter(item, files)))
            return result
        if type(value) in (dict, MappingProxyType):
            result = []
            for key, item in value.items():
                result.append([(yield from self.encode_iter(key, files)),
                               (yield from self.encode_iter(item, files))])
            return {'$dict': result}
        raise TypeError('Unsupported execution payload: ' + type(value).__name__)

    def encode(self, value, files):
        generator = self.encode_iter(value, files)
        while True:
            try:
                next(generator)
            except StopIteration as finished:
                return finished.value

    def decode(self, value, *, copy=False):
        if isinstance(value, list):
            return [self.decode(v, copy=copy) for v in value]
        if not isinstance(value, dict):
            return value
        if '$array' in value:
            name = value['$array']
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.startswith('array-') or not name.endswith('.npy')):
                raise ValueError('Invalid payload path')
            path = self.path / name
            if path.is_symlink():
                raise ValueError('Payload symlinks are forbidden')
            dtype = np.dtype(value['dtype'])
            shape = tuple(value['shape'])
            if dtype.hasobject or dtype.kind not in 'bifu' or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('Invalid array descriptor')
            size = math.prod(shape) * dtype.itemsize
            if size != value['bytes'] or size > self.maximum:
                raise ValueError('Invalid array length')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.dtype != dtype or array.shape != shape or array.nbytes != size:
                array._mmap.close()
                raise ValueError('Array descriptor does not match payload')
            if copy:
                result = np.array(array, copy=True)
                array._mmap.close()
                result.flags.writeable = False
                return result
            self.owned.add(path)
            self.mapped.setdefault(path, []).append(weakref.ref(array))
            array.flags.writeable = False
            return array
        if '$tuple' in value:
            return tuple(self.decode(v, copy=copy) for v in value['$tuple'])
        if '$dict' in value:
            return {self.decode(k, copy=copy): self.decode(v, copy=copy) for k, v in value['$dict']}
        if '$float' in value and value['$float'] in ('nan', 'inf', '-inf'):
            return float(value['$float'])
        raise ValueError('Invalid payload record')

    def release_files(self, names):
        for name in names:
            path = self.path / name
            if Path(name).name != name or not name.startswith('array-'):
                raise ValueError('Invalid owned payload')
            try:
                path.unlink(missing_ok=True)
                self.owned.discard(path)
                self.mapped.pop(path, None)
            except OSError:
                # Windows prevents unlinking a still-open mapping. Retry later.
                pass

    def collect(self):
        for path, references in tuple(self.mapped.items()):
            live = [ref for ref in references if ref() is not None]
            if live:
                self.mapped[path] = live
            else:
                self.release_files([path.name])

    def close(self):
        # NumPy owns each mapping's lifetime. Forcibly closing a live view can
        # segfault a caller holding its array. POSIX permits unlinking those
        # files; Windows retains them and the journal correctly stays unclean
        # until the owning process exits and recovery can reclaim them.
        self.mapped.clear()
        self.release_files([p.name for p in tuple(self.owned)])
