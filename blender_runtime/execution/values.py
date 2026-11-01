"""Explicit immutable numerical snapshots for retained execution values.

Model records stay in extension-specific codecs; exact built-in containers and
plain numerical arrays form the shared boundary.
"""
from types import MappingProxyType
import numpy as np


def seal(value):
    active, memo = set(), {}

    def visit(item):
        kind = type(item)
        if item is None or kind in (bool, str, int, float):
            return item
        if isinstance(item, np.generic):
            if item.dtype.hasobject or item.dtype.kind not in 'bifu':
                raise TypeError('Execution scalars must be numerical')
            return visit(item.item())
        identity = id(item)
        if identity in active:
            raise ValueError('Execution values cannot contain container cycles')
        if identity in memo:
            return memo[identity]
        active.add(identity)
        try:
            if isinstance(item, np.ndarray):
                result = np.array(item, copy=True, order='C', subok=False)
                if result.dtype.hasobject or result.dtype.kind not in 'bifu' or result.dtype.metadata is not None:
                    raise TypeError('Execution arrays must have a numerical dtype')
                result.flags.writeable = False
            elif kind in (dict, MappingProxyType):
                result = MappingProxyType({visit(key): visit(entry) for key, entry in item.items()})
            elif kind in (tuple, list):
                result = tuple(visit(entry) for entry in item)
            else:
                raise TypeError('Unsupported retained execution value: ' + kind.__name__)
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    return visit(value)
