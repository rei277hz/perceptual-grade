"""Explicit execution machine dependencies, supplied by an App."""
from dataclasses import dataclass

from ..machine import Machine


@dataclass(frozen=True, slots=True)
class ExecutionMachines:
    attempt: Machine
    gpu: Machine
    retry: Machine
    startup: Machine
    coordinator: Machine | None = None

    def __post_init__(self):
        if self.coordinator is None:
            from ..machines import coordinator_machine
            object.__setattr__(self, 'coordinator', coordinator_machine('execution'))
        if any(type(machine) is not Machine for machine in (self.attempt, self.gpu, self.retry, self.startup, self.coordinator)):
            raise TypeError('Execution machines must be typed machine descriptors')

    @staticmethod
    def event(machine, **parameters):
        return machine.event_type(**parameters)
