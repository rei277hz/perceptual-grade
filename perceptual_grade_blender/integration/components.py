"""Explicit session component exports, retaining original lexical state."""


class Component:
    def __init__(self, exports=None, *, state=None, readers=None, writers=None):
        object.__setattr__(self, '_Component__exports', {} if exports is None else dict(exports))
        object.__setattr__(self, '_Component__state', state)
        object.__setattr__(self, '_Component__readers', {} if readers is None else dict(readers))
        object.__setattr__(self, '_Component__writers', {} if writers is None else dict(writers))

    def __getattr__(self, name):
        state = object.__getattribute__(self, '_Component__state')
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        readers = object.__getattribute__(self, '_Component__readers')
        if name in readers:
            return readers[name]()
        exports = object.__getattribute__(self, '_Component__exports')
        if name in exports:
            return exports[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        state = object.__getattribute__(self, '_Component__state')
        if state is not None and hasattr(state, name):
            setattr(state, name, value)
            return
        writers = object.__getattribute__(self, '_Component__writers')
        if name in writers:
            writers[name](value)
            exports = object.__getattribute__(self, '_Component__exports')
            if name in exports:
                exports[name] = value
            return
        raise AttributeError('Session component export is read-only: ' + name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise KeyError(name) from error

    def __setitem__(self, name, value):
        setattr(self, name, value)


def component_get(component, name, default=None):
    """Read an export without reserving a product function name."""
    return getattr(component, name, default)


def component_items(component):
    exports = object.__getattribute__(component, '_Component__exports')
    readers = object.__getattribute__(component, '_Component__readers')
    state = object.__getattribute__(component, '_Component__state')
    names = exports.keys() | readers.keys()
    if state is not None:
        names |= state.__dict__.keys()
    return ((name, getattr(component, name)) for name in names)


def set_component_metadata(component, **metadata):
    exports = object.__getattribute__(component, '_Component__exports')
    for name, value in metadata.items():
        exports.setdefault(name, value)


class ComponentRef:
    """Lazy peer reference; factory imports do not instantiate a peer early."""
    def __init__(self, session, name):
        object.__setattr__(self, '_ComponentRef__session', session)
        object.__setattr__(self, '_ComponentRef__name', name)

    def __getattr__(self, name):
        return getattr(self.__session._component(self.__name), name)

    def __setattr__(self, name, value):
        setattr(self.__session._component(self.__name), name, value)
