"""Actor workflow admissions with explicit, captured receipt identities."""


class Protocol:
    def __init__(self, application, name, *, initial=None, identity=None):
        self.store = application.app.store(machine=application.machines[name + '_coordinator'],
                                           initial=initial or {})
        self.module = self.store.machine.module
        # Only foreground code updates these values. Delayed work receives an
        # immutable Envelope from receipt(), never defaults from the current state.
        self.identity = dict(identity or {})

    @property
    def state(self):
        return self.store.state

    def receipt(self, kind, **values):
        return self.store.event(self.module.Event(kind=kind, **(self.identity | values)))

    def dispatch(self, receipt):
        dispatch = self.store._dispatch_cleanup if self.store.app.fenced else self.store.dispatch
        return dispatch(receipt)

    def event(self, kind, **values):
        # Cleanup may follow App admission fencing; these are foreground events,
        # not receipts relabeled from a newer actor snapshot.
        event = self.module.Event(kind=kind, **(self.identity | values))
        return self.dispatch(event)

    def require(self, kind, action, **values):
        effect = self.event(kind, **values)
        if not effect.accepted or effect.action != action:
            raise RuntimeError(f'{self.module.__name__} rejected {kind}: {effect.action}')
        return effect

    def close(self):
        self.store.close()
