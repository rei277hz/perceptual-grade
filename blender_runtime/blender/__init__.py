"""Optional Blender adapters; importing this package does not import bpy."""
from dataclasses import dataclass

from .native import NativeAdapter
from .registration import Registration


@dataclass(frozen=True, slots=True)
class BlenderConfig:
    native: bool = False

    def __post_init__(self):
        if type(self.native) is not bool:
            raise TypeError('Native adapter configuration must be Boolean')


class BlenderContext:
    def __init__(self, app, config):
        self.app = app
        self.registration = Registration(app)
        if config.native and app.ownership is None:
            raise ValueError('Native removal requires the App ownership capability')
        self.native = NativeAdapter(app.ownership) if config.native else None

    def close(self):
        self.registration.close()


__all__ = ['BlenderConfig', 'BlenderContext', 'NativeAdapter', 'Registration']
