"""Bootstrap a private component namespace without registering an extension."""
from importlib import import_module
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
from types import ModuleType


def main():
    config = json.loads(Path(sys.argv[-1]).read_text())
    package = Path(config['package'])
    name = '_extension_helper_' + config['session']
    # The caller passes the component namespace tree: color_tools during
    # development, or this extension's root when installed.
    # Do not execute any extension initializer or publish a global alias.
    module = ModuleType(name)
    module.__path__ = [str(package)]
    module.__package__ = name
    module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    import_module(name + '.blender_runtime.execution.worker').run(config, name)


if __name__ == '__main__':
    main()
