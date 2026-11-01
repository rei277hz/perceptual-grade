"""Explicit Blender host configuration for an App's lazy execution service."""
import os
from pathlib import Path
import sys

from ..execution.runtime import ExecutionRuntime
from ..execution.values import seal as seal_values


def launch_command(config):
    import bpy
    if bpy.app.binary_path:
        return [bpy.app.binary_path, '--background', '--factory-startup', '--disable-autoexec',
                '--python-exit-code', '1', '--python', config['worker_entry'], '--', config['config_path']]
    return [sys.executable, config['worker_entry'], config['config_path']]


def create_runtime(app, *, machines, package, profile, seal=seal_values, limits=None):
    import bpy
    app.require_open()
    root = bpy.utils.user_resource('CONFIG', path=app.storage_name + '/execution', create=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(
        p for p in sys.path if p and '/bpy/' not in p.replace('\\', '/'))
    settings = {'backend': getattr(bpy.context.preferences.system, 'gpu_backend', '')}
    return ExecutionRuntime(root, app=app, machines=machines, package=Path(package), profile=profile,
                            seal=seal, limits=limits, launcher=launch_command,
                            worker_host='blender_runtime.blender.worker', host_settings=settings,
                            environment=environment)
