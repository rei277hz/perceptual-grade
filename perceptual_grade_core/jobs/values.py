"""Copy numerical values before a persistent execution owner can retain them.

Only exact built-in containers and scalar types enter this boundary. Arrays
receive fresh, plain, owned storage: a numerical dtype alone does not exclude
RNA hidden in a subclass attribute or an arbitrary backing object's fields.
"""
from types import MappingProxyType
from dataclasses import fields

import numpy as np


def seal(value):
    """Return an immutable-container snapshot with no arbitrary object edges."""
    active = set()
    memo = {}

    # Dataclasses cross the helper boundary only when they are part of the
    # named scope identity or one of the immutable numerical parameter
    # records.  Arbitrary dataclasses can retain Blender/RNA objects and stay
    # rejected by this explicit allowlist.
    from ..scopes.identity import ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity, validate_identity
    from ..model import tone, color, linear
    allowed = (ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity,
               tone.Parameters, color.Parameters, linear.Parameters,
               linear.GradeSpace)

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
            if type(item) in allowed:
                if type(item) in (ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity):
                    validate_identity(item)
                result = type(item)(**{
                    field.name: visit(getattr(item, field.name))
                    for field in fields(item)})
            elif isinstance(item, np.ndarray):
                result = np.array(item, copy=True, order='C', subok=False)
                # Inspect the plain copy: subclasses can override `dtype`.
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
