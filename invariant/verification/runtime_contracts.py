"""Runtime checks generated from verified requirements; no solver dependency."""

import ast
from dataclasses import replace
import inspect
import operator


class Contract:
    def __init__(self, module, requirements, schema):
        self.module = module
        self.requirements = requirements
        self.schema = schema
        self.parsed = {}

    def expression(self, text, env):
        if text not in self.parsed:
            self.parsed[text] = ast.parse(text, mode="eval").body
        return self._expr(self.parsed[text], env)

    def _expr(self, node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.schema["constants"]:
                return self.schema["constants"][node.id]
            raise ValueError("Unknown contract variable: " + node.id)
        if isinstance(node, ast.Attribute):
            value = self._expr(node.value, env)
            name = type(value).__name__
            if name not in self.schema["records"] or node.attr not in self.schema["records"][name]:
                raise ValueError("Invalid contract attribute")
            return getattr(value, node.attr)
        if isinstance(node, ast.Tuple):
            return tuple(self._expr(x, env) for x in node.elts)
        if isinstance(node, ast.Subscript):
            return self._expr(node.value, env)[self._expr(node.slice, env)]
        if isinstance(node, ast.UnaryOp):
            function = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}.get(type(node.op))
            if function is None:
                raise ValueError("Invalid unary operator")
            return function(self._expr(node.operand, env))
        if isinstance(node, ast.BinOp):
            function = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}.get(type(node.op))
            if function is None:
                raise ValueError("Invalid binary operator")
            return function(self._expr(node.left, env), self._expr(node.right, env))
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self._expr(item, env)
                if type(value) is not bool:
                    raise ValueError("Expected Boolean contract expression")
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self._expr(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._expr(comparator, env)
                function = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}.get(type(op))
                if function is None:
                    raise ValueError("Invalid comparison")
                if not function(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._expr(node.body if self._expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ('forall_range', 'exists_range', 'sum_range'):
                if len(node.args) != 4 or node.keywords or not isinstance(node.args[0], ast.Name) or node.args[0].id in env:
                    raise ValueError('Invalid bounded binder')
                lower, upper = (self._expr(x, env) for x in node.args[1:3])
                if type(lower) is not int or type(upper) is not int:
                    raise ValueError('Bounded binder endpoints must be integers')
                values = (self._expr(node.args[3], env | {node.args[0].id: index}) for index in range(lower, upper))
                def checked():
                    for value in values:
                        if type(value) is not (int if name == 'sum_range' else bool):
                            raise ValueError('Invalid bounded binder body type')
                        yield value
                return {'forall_range': all, 'exists_range': any, 'sum_range': sum}[name](checked())
            if name == "replace":
                function = replace
            elif name == 'len':
                function = len
            elif name in self.schema["functions"] or name in self.schema["records"]:
                function = getattr(self.module, name)
            else:
                raise ValueError("Invalid contract call")
            return function(*[self._expr(x, env) for x in node.args], **{x.arg: self._expr(x.value, env) for x in node.keywords})
        raise ValueError("Unsupported contract expression")

    def typed(self, value, typ):
        kind, args = typ["kind"], typ["args"]
        if kind == "int":
            return type(value) is int
        if kind == "bool":
            return type(value) is bool
        if kind == 'str':
            return type(value) is str
        if kind == 'union':
            return any(self.typed(value, member) for member in args)
        if kind == "literal":
            return any(type(value) is type(x) and value == x for x in args)
        if kind == "tuple":
            return type(value) is tuple and len(value) == len(args) and all(self.typed(v, t) for v, t in zip(value, args))
        if kind == 'sequence':
            return type(value) is tuple and all(self.typed(v, args[0]) for v in value)
        if kind == "record":
            return type(value) is getattr(self.module, args[0]) and all(self.typed(getattr(value, key), t) for key, t in self.schema["records"][args[0]].items())
        raise ValueError("Unknown generated type")

    def _arguments(self, name, arguments):
        if name not in self.schema["functions"]:
            raise ValueError("Unknown contract function: " + name)
        try:
            values = inspect.signature(getattr(self.module, name)).bind(**arguments)
            values.apply_defaults()
            return dict(values.arguments)
        except TypeError as error:
            raise ValueError(str(error)) from error

    def _clauses(self, expressions, env):
        for expression in expressions:
            try:
                result = self.expression(expression, env)
            except (TypeError, AttributeError, IndexError, KeyError, ArithmeticError) as error:
                raise ValueError("Contract expression is undefined: " + expression) from error
            if result is not True:
                raise ValueError("Contract failed: " + expression)

    def validate_before(self, name, arguments):
        env = self._arguments(name, arguments)
        for key, typ in self.schema["functions"][name]["arguments"].items():
            if not self.typed(env[key], typ):
                raise ValueError("Invalid argument type: " + name + "." + key)
        self._clauses(self.requirements.get("contracts", {}).get(name, {}).get("requires", []), env)
        if name == self.requirements["step"]:
            self.validate_state(next(iter(env.values())))

    def validate_after(self, name, arguments, result):
        env = self._arguments(name, arguments)
        if not self.typed(result, self.schema["functions"][name]["returns"]):
            raise ValueError("Invalid return type: " + name)
        self._clauses(self.requirements.get("contracts", {}).get(name, {}).get("ensures", []), {**env, "result": result})
        if name == self.requirements["step"]:
            self.validate_state(result[0])
        if name in self.requirements["initializers"]:
            self.validate_state(result)

    def validate_state(self, state):
        if not self.typed(state, {"kind": "record", "args": [self.requirements["state"]]}):
            raise ValueError("Invalid state type")
        self._clauses(self.requirements["invariants"].values(), {"state": state})


def bind(module, requirements, schema):
    return Contract(module, requirements, schema)
