from dataclasses import fields, is_dataclass
import asyncio
import inspect
import threading
from types import MappingProxyType, ModuleType, UnionType
from typing import Literal, Union, get_args, get_origin


class ContractError(ValueError):
    pass


class AdmissionError(ContractError):
    """Rejected input has not executed or committed a transition."""


def _annotation(annotation, ancestry=()):
    """Validate concrete immutable domains without evaluating annotations."""
    if annotation in (int, bool, str):
        return
    if annotation in ancestry:
        raise TypeError('recursive record domains are unsupported')
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType) and len(arguments) == 2 and set(arguments) == {int, str}:
        return
    if origin is Literal and arguments and len({type(item) for item in arguments}) == 1 and all(type(item) in (int, bool, str) for item in arguments):
        return
    if origin is tuple:
        if not arguments and annotation != tuple[()]:
            raise TypeError('tuple domains require an explicit shape')
        for item in arguments[:1] if len(arguments) == 2 and arguments[1] is Ellipsis else arguments:
            _annotation(item, ancestry)
        return
    if isinstance(annotation, type) and is_dataclass(annotation) and annotation.__dataclass_params__.frozen:
        for field in fields(annotation):
            _annotation(field.type, ancestry + (annotation,))
        return
    raise TypeError(f'unsupported immutable annotation: {annotation!r}')


def validate(value, annotation, path='value'):
    origin, arguments = get_origin(annotation), get_args(annotation)
    if annotation in (int, bool, str):
        valid = type(value) is annotation
    elif origin in (Union, UnionType) and len(arguments) == 2 and set(arguments) == {int, str}:
        valid = type(value) in (int, str)
    elif origin is Literal:
        valid = any(type(value) is type(item) and value == item for item in arguments)
    elif origin is tuple:
        sequence = len(arguments) == 2 and arguments[1] is Ellipsis
        valid = type(value) is tuple and (sequence or len(value) == len(arguments))
        if valid:
            for index, item in enumerate(value):
                validate(item, arguments[0] if sequence else arguments[index], f'{path}[{index}]')
    elif isinstance(annotation, type) and is_dataclass(annotation):
        valid = type(value) is annotation and annotation.__dataclass_params__.frozen
        if valid:
            for field in fields(annotation):
                validate(getattr(value, field.name), field.type, f'{path}.{field.name}')
    else:
        raise ContractError(f'unsupported annotation for {path}: {annotation!r}')
    if not valid:
        raise ContractError(f'{path} does not match {annotation!r}')


class Machine:
    def __init__(self, module, *, contracts=None, initializers=('new',), queries=(), default_initializer=None):
        if not isinstance(module, ModuleType):
            raise TypeError('machine requires its explicit module')
        if (type(initializers) is not tuple or not initializers or type(queries) is not tuple
                or any(type(name) is not str for name in (*initializers, *queries))
                or len(set(initializers)) != len(initializers) or len(set(queries)) != len(queries)):
            raise TypeError('machine bindings require explicit unique name tuples')
        self.module = module
        if contracts is None:
            self.contracts = None
        elif 'binder' in inspect.signature(contracts.bind).parameters:
            from .verification.runtime_contracts import bind
            self.contracts = contracts.bind(module, binder=bind)
        else:
            self.contracts = contracts.bind(module)
        self.initializers = frozenset(initializers)
        self.queries = frozenset(queries)
        if default_initializer is not None and default_initializer not in initializers:
            raise TypeError('default initializer must be explicitly declared')
        self.default_initializer = initializers[0] if default_initializer is None else default_initializer
        self.state_type = inspect.signature(module.step).parameters['state'].annotation
        self.event_type = inspect.signature(module.step).parameters['event'].annotation
        functions = {}
        for name in (*initializers, *queries, 'step'):
            function = getattr(module, name)
            if not inspect.isfunction(function) or function.__globals__ is not module.__dict__:
                raise ContractError(f'{name} must be a function from the bound module')
            signature = inspect.signature(function)
            for parameter in signature.parameters.values():
                if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                    raise TypeError('machine functions require named parameters')
                _annotation(parameter.annotation)
            _annotation(signature.return_annotation)
            functions[name] = function
        if tuple(inspect.signature(module.step).parameters) != ('state', 'event'):
            raise TypeError('step requires state and event parameters')
        result = get_args(inspect.signature(module.step).return_annotation)
        if (get_origin(inspect.signature(module.step).return_annotation) is not tuple or len(result) != 2
                or result[0] is not self.state_type or not is_dataclass(self.state_type)
                or not is_dataclass(self.event_type) or not is_dataclass(result[1])):
            raise TypeError('step must return its state and frozen effects records')
        if any(inspect.signature(functions[name]).return_annotation is not self.state_type for name in initializers):
            raise TypeError('initializers must return the machine state')
        self.effects_type = result[1]
        self.step = functions['step']
        self._functions = MappingProxyType(functions)
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            from dataclasses import FrozenInstanceError
            raise FrozenInstanceError('machine bindings are immutable')
        object.__setattr__(self, name, value)

    def call(self, name, **arguments):
        if name not in self._functions:
            raise AdmissionError('function is not declared: ' + str(name))
        function = self._functions[name]
        try:
            signature = inspect.signature(function)
            bound = signature.bind(**arguments)
            bound.apply_defaults()
            for key, value in bound.arguments.items():
                validate(value, signature.parameters[key].annotation, key)
            if self.contracts:
                self.contracts.validate_before(name, dict(bound.arguments))
        except (ValueError, TypeError) as error:
            raise AdmissionError(str(error)) from error
        try:
            result = function(**bound.arguments)
            validate(result, signature.return_annotation, f'{name} result')
            if self.contracts:
                self.contracts.validate_after(name, dict(bound.arguments), result)
            return result
        except ValueError as error:
            raise ContractError(str(error)) from error

    def validate_state(self, state):
        validate(state, self.state_type, 'state')
        if self.contracts:
            try:
                self.contracts.validate_state(state)
            except ValueError as error:
                raise ContractError(str(error)) from error


class Store:
    def __init__(self, machine, *, initializer='new', arguments=None, owner_task=None):
        if initializer not in machine.initializers:
            raise AdmissionError('initializer is not declared')
        self.machine = machine
        self.owner = threading.get_ident()
        self._owner_task = owner_task
        self._check_owner()
        self._state = machine.call(initializer, **(arguments or {}))
        machine.validate_state(self._state)

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('store access must remain on its coordinator thread')
        if self._owner_task is not None:
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if current is not self._owner_task:
                raise RuntimeError('store access must remain in its owning actor task')

    @property
    def state(self):
        self._check_owner()
        return self._state

    def preview(self, event):
        self._check_owner()
        result = self.machine.call('step', state=self._state, event=event)
        self.machine.validate_state(result[0])
        return result

    def dispatch(self, event):
        state, effects = self.preview(event)
        self._state = state
        return effects

    def query(self, name, **arguments):
        self._check_owner()
        if name not in self.machine.queries:
            raise AdmissionError('query is not declared')
        if 'state' in inspect.signature(self.machine._functions[name]).parameters:
            if 'state' in arguments:
                raise AdmissionError('store queries use their current state')
            arguments = dict(arguments, state=self._state)
        return self.machine.call(name, **arguments)
