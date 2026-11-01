"""Compose explicit product capabilities without starting a helper."""
from pathlib import Path

from ..blender_runtime import App, ExecutionConfig
from ..blender_runtime.blender import BlenderConfig
from ..blender_runtime.blender.execution import create_runtime
from ..blender_runtime.blender.native import keys
from ..blender_runtime.execution.machines import ExecutionMachines
from ..blender_runtime.ownership import OwnershipConfig
from ..blender_runtime.machines import coordinator_machine
from .jobs.configuration import storage_name, profile, seal
from .integration.components import Component


def create_app(application):
    package = Path(__file__).absolute().parent.parent
    machines = ExecutionMachines(*(application.machines[name] for name in ('attempt', 'gpu', 'retry', 'startup')))
    app = App(package=package, storage_name=storage_name,
               execution=ExecutionConfig(lambda app: create_runtime(app, machines=machines,
                   package=package, profile=profile, seal=seal)),
               ownership=OwnershipConfig(application.machines['ownership'], keys=keys,
                   require_clean=lambda: application.component('ownership.transactions').require_clean()),
               blender=BlenderConfig(native=True), lifecycle=coordinator_machine('lifecycle'))
    def retired():
        capture = application.components.get('scopes.capture')
        if capture is not None and not capture.collect_retired():
            raise RuntimeError('Native capture cleanup has not retired')
        bake = application.components.get('graph.bake')
        if bake is not None and not bake.drain_committed():
            raise RuntimeError('Committed bake cleanup has not retired')
        jobs = application.components.get('scopes.jobs')
        if jobs is not None and not jobs.drain_retired():
            raise RuntimeError('Scope analysis cleanup has not retired')
        lut = application.components.get('graph.lut')
        if lut is not None and not lut.collect_retired():
            raise RuntimeError('LUT cleanup has not retired')
    app.on_retired(retired)
    return app


def create(_application):
    return Component({'get_runtime': _application.app.execution.get_runtime,
                      'current_runtime': _application.app.execution.current_runtime,
                      '__name__': __name__})
