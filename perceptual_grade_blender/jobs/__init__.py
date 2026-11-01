"""Shared job execution. Importing this package never starts helpers."""

from ..integration.components import Component

def create(_application):


    def get_runtime():
        obtain = _application.component('services').get_runtime
        return obtain()

    def _set_export_get_runtime(value):
        nonlocal get_runtime
        get_runtime = value

    return Component(readers={
        "__name__": lambda: __name__,
        'get_runtime': lambda: get_runtime,
    }, writers={
    'get_runtime': _set_export_get_runtime})
