"""Blender-independent request and event contracts."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
from typing import Any

PROTOCOL = 1


@dataclass(frozen=True)
class JobSpec:
    operation: str
    inputs: dict
    lane: str = 'bulk'
    priority: int = 0
    key: Any = None


@dataclass(frozen=True)
class JobEvent:
    handle: int
    kind: str
    value: Any = None


@dataclass(frozen=True)
class RuntimeLimits:
    pending_jobs: int = 64
    payload_bytes: int = 512 * 1024 * 1024
    result_bytes: int = 128 * 1024 * 1024
    startup_seconds: float = 60.
    stalled_seconds: float = 300.
    shutdown_seconds: float = 5.


class ExecutionError(RuntimeError):
    pass


class CapacityError(ExecutionError):
    pass


@dataclass(frozen=True)
class JobFailure:
    """A value receipt cannot retain a caller through an exception traceback."""
    message: str
    cancelled: bool = False

    def exception(self):
        return (CancelledError if self.cancelled else ExecutionError)(self.message)


@dataclass(frozen=True)
class ShutdownReport:
    clean: bool
    errors: tuple = ()


@dataclass(frozen=True)
class ExecutionDiagnostic:
    operation: str
    reason: str
    traceback: str
    helper_log: str
    backend: str
    device: str
    fingerprint: str
    session: str
    attempt: int
    generation: int
    pid: int = 0
    helper_log_path: str = ''
    diagnostic_path: str = ''
    role: str = 'gpu'


@dataclass(frozen=True)
class RuntimeProfile:
    """Per-package execution policy, copied into each helper configuration."""
    lanes: tuple = ('bulk', 'gpu')
    gpu_probes: tuple = (('gpu_probe', 'gpu'),)
    fallback_lane: str = 'bulk'
    uncached_operations: tuple = ()
    dispatcher: str = 'operations'

    def __post_init__(self):
        if (type(self.lanes) is not tuple or not self.lanes
                or any(type(lane) is not str or not lane.isidentifier() for lane in self.lanes)
                or len(set(self.lanes)) != len(self.lanes)):
            raise ValueError('Execution lanes must be an immutable tuple of distinct identifiers')
        if (type(self.gpu_probes) is not tuple
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(item) is not str or not item for item in pair)
                       for pair in self.gpu_probes)):
            raise ValueError('GPU probes must be immutable operation/lane pairs')
        probes = dict(self.gpu_probes)
        if len(probes) != len(self.gpu_probes) or any(role not in self.lanes for role in probes.values()):
            raise ValueError('GPU probes must identify declared execution lanes')
        if type(self.fallback_lane) is not str or self.fallback_lane not in self.lanes or self.fallback_lane in probes.values():
            raise ValueError('GPU fallback must identify a CPU lane')
        if (type(self.uncached_operations) is not tuple
                or any(type(operation) is not str or not operation for operation in self.uncached_operations)
                or len(set(self.uncached_operations)) != len(self.uncached_operations)):
            raise ValueError('Uncached operations must be an immutable tuple of distinct names')
        if type(self.dispatcher) is not str or not self.dispatcher or not all(part.isidentifier() for part in self.dispatcher.split('.')):
            raise ValueError('Worker dispatcher must be a package-relative module name')
