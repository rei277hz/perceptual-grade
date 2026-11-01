"""Value-only host receipts for shared actor-owned coordinator policies."""
from dataclasses import fields


class Protocol:
    def __init__(self, app, machine):
        self.app = app
        self.store = app.store(machine=machine)
        self.module = machine.module
        self._fields = tuple(field.name for field in fields(machine.event_type))

    @property
    def state(self):
        return self.store.state

    def event(self, kind, **values):
        state = self.state
        identity = {name: getattr(state, name) for name in self._fields
                    if name != 'kind' and hasattr(state, name)}
        identity.update(values)
        return self.store._dispatch_cleanup(self.module.Event(kind=kind, **identity))

    def require(self, kind, action, **values):
        effect = self.event(kind, **values)
        if not effect.accepted or effect.action != action:
            raise RuntimeError(f'Coordinator rejected {kind}: expected {action}, got {effect.action}')
        return effect
