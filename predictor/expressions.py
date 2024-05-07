from __future__ import annotations

import abc
import enum
import itertools
import typing

NamespaceType = typing.MutableMapping[str, typing.Any]
ExpressionTypes: typing.Set[typing.Type[typing.Any]] = set()


def define_expression(t: typing.Type[typing.Any]):
    ExpressionTypes.add(t)
    return t


def is_expression(o: typing.Any) -> bool:
    return type(o) in ExpressionTypes


class Context(enum.Enum):
    Load = enum.auto()
    Store = enum.auto()
    Delete = enum.auto()


class EvaluateHook(typing.Protocol):
    def __call__(self, expression: Expression, operator: typing.Any, *args, **kwargs) -> typing.Tuple[bool, typing.Any]:
        pass


@define_expression
class Expression(abc.ABC):
    def __init__(self):
        self.__attributes = {}

    # used for hashing the expression itself, using call(hash)(target) to get the hash expression
    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    # unary arithmetic symbols

    def __invert__(self) -> UnaryOperation:
        return UnaryOperation(self, UnaryOperator.INVERT)

    def __neg__(self) -> UnaryOperation:
        return UnaryOperation(self, UnaryOperator.NEGATIVE)

    def __pos__(self) -> UnaryOperation:
        return UnaryOperation(self, UnaryOperator.POSITIVE)

    # binary arithmetic symbols
    def __add__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.ADD)

    def __radd__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.ADD)

    def __and__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.BITWISE_AND)

    def __rand__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.BITWISE_AND)

    def __eq__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.EQUAL)

    def __req__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.EQUAL)

    def __floordiv__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.FLOOR_DIVIDE)

    def __rfloordiv__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.FLOOR_DIVIDE)

    def __ge__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.GREATER_EQUAL)

    def __rge__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.GREATER_EQUAL)

    def __gt__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.GREATER_THAN)

    def __rgt__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.GREATER_THAN)

    def __lshift__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.LEFT_SHIFT)

    def __rlshift__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.LEFT_SHIFT)

    def __le__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.LESS_EQUAL)

    def __rle__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.LESS_EQUAL)

    def __lt__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.LESS_THAN)

    def __rlt__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.LESS_THAN)

    def __matmul__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.MATRIX_MULTIPLY)

    def __rmatmul__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.MATRIX_MULTIPLY)

    def __mod__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.MODULO)

    def __rmod__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.MODULO)

    def __mul__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.MULTIPLY)

    def __rmul__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.MULTIPLY)

    def __ne__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.NOT_EQUAL)

    def __rne__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.NOT_EQUAL)

    def __or__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.BITWISE_OR)

    def __ror__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.BITWISE_OR)

    def __pow__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.POWER)

    def __rpow__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.POWER)

    def __rshift__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.RIGHT_SHIFT)

    def __rrshift__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.RIGHT_SHIFT)

    def __sub__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.SUBTRACT)

    def __rsub__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.SUBTRACT)

    def __truediv__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.TRUE_DIVIDE)

    def __rtruediv__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.TRUE_DIVIDE)

    def __xor__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(self, other, BinaryOperator.BITWISE_XOR)

    def __rxor__(self, other: MaybeExpression) -> BinaryOperation:
        return BinaryOperation(other, self, BinaryOperator.BITWISE_XOR)

    def __call__(self, *args, **kwargs) -> FunctionCall:
        return FunctionCall(self, args, kwargs)

    def __contains__(self, item) -> Contains:
        return Contains(self, item)

    def __getattr__(self, item: str) -> Attribute:
        attributes = self.__attributes
        if item not in attributes:
            attributes[item] = Attribute(self, item, Context.Load)
        return attributes[item]

    def __getitem__(self, item) -> Item:
        return Item(self, item, Context.Load)

    def attribute(self, name: str) -> Attribute:
        if name not in self.__attributes:
            self.__attributes[name] = Attribute(self, name, Context.Load)
        return self.__attributes[name]

    # used for equality checks, since __eq__ is used for binary operations
    @abc.abstractmethod
    def equal(self, other: Expression):
        pass

    @abc.abstractmethod
    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        pass

    @abc.abstractmethod
    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        pass

    @abc.abstractmethod
    def match(self, expression: Expression) -> bool:
        pass

    @abc.abstractmethod
    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        pass


MaybeExpression = typing.Union[Expression, typing.Any]


def equal(lhs: MaybeExpression, rhs: MaybeExpression) -> bool:
    if lhs is rhs:
        return True
    isLhsExpression = is_expression(lhs)
    isRhsExpression = is_expression(rhs)
    if isLhsExpression and isRhsExpression:
        result = lhs.equal(rhs)
        if result is NotImplemented:
            result = rhs.equal(lhs)
        if result is NotImplemented:
            raise TypeError(f"Cannot compare {lhs} and {rhs}")
        return result
    if isLhsExpression or isRhsExpression:
        return False
    return lhs == rhs


@define_expression
class Parameter(Expression):
    def __init__(self, index: int | str):
        super().__init__()
        self.__index = index
        self.__hash = hash((Parameter, self.__index))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Parameter:
            return False
        return self.__index == other.__index

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        raise NotImplementedError()

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        leaves.add(self)
        return leaves

    def match(self, expression: Expression) -> bool:
        return bool(self == expression)

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        return self


@define_expression
class Parameters:
    def __getitem__(self, item):
        match item:
            case int() | str():
                return Parameter(item)
            case _:
                raise TypeError()


p = Parameters()


@define_expression
class Return(Expression):
    def __init__(self):
        super().__init__()
        self.__hash = hash(Return)

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression):
        return type(other) is Return

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        raise NotImplementedError()

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        leaves.add(self)
        return leaves

    def match(self, expression: Expression) -> bool:
        return bool(self == expression)

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        raise NotImplementedError()


r = Return()


@define_expression
class Attribute(Expression):
    def __init__(self, target: MaybeExpression, name: str, context: Context):
        super().__init__()
        self.__target = target
        self.__name = name
        self.__context = context
        self.__hash = hash((Attribute, self.__target, self.__name))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Attribute:
            return False
        return equal(self.__target, other.__target) and self.__name == other.__name

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        target = self.__target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__target) else self.__target
        match self.__context:
            case Context.Load:
                if hook is not None:
                    overwrite, result = hook(self, None, target)
                    if overwrite:
                        return result
                return getattr(target, self.__name)
            case Context.Store:
                setattr(target, self.__name, assignment)
            case Context.Delete:
                delattr(target, self.__name)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            return self.__target.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return self.__target.match(expression) if is_expression(self.__target) else False

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        result = Attribute(target, self.__name, self.__context)
        return substitution[result] if result in substitution else result


@enum.unique
class BinaryOperator(enum.Enum):
    ADD = enum.auto(),
    BITWISE_AND = enum.auto(),
    BITWISE_OR = enum.auto(),
    BITWISE_XOR = enum.auto()
    EQUAL = enum.auto(),
    FLOOR_DIVIDE = enum.auto(),
    GREATER_THAN = enum.auto(),
    GREATER_EQUAL = enum.auto(),
    IS = enum.auto(),
    IS_NOT = enum.auto(),
    LEFT_SHIFT = enum.auto(),
    LESS_EQUAL = enum.auto(),
    LESS_THAN = enum.auto(),
    LOGICAL_AND = enum.auto(),
    LOGICAL_OR = enum.auto(),
    MATRIX_MULTIPLY = enum.auto(),
    MODULO = enum.auto(),
    MULTIPLY = enum.auto(),
    NOT_EQUAL = enum.auto(),
    POWER = enum.auto(),
    RIGHT_SHIFT = enum.auto(),
    SUBTRACT = enum.auto(),
    TRUE_DIVIDE = enum.auto(),

    def apply(self, lhs: typing.Any, rhs: typing.Any) -> typing.Any:
        match self:
            case BinaryOperator.ADD:
                return lhs + rhs
            case BinaryOperator.BITWISE_AND:
                return lhs & rhs
            case BinaryOperator.BITWISE_OR:
                return lhs | rhs
            case BinaryOperator.BITWISE_XOR:
                return lhs ^ rhs
            case BinaryOperator.EQUAL:
                return lhs == rhs
            case BinaryOperator.FLOOR_DIVIDE:
                return lhs // rhs
            case BinaryOperator.GREATER_EQUAL:
                return lhs >= rhs
            case BinaryOperator.GREATER_THAN:
                return lhs > rhs
            case BinaryOperator.IS:
                return lhs is rhs
            case BinaryOperator.IS_NOT:
                return lhs is not rhs
            case BinaryOperator.LEFT_SHIFT:
                return lhs << rhs
            case BinaryOperator.LESS_EQUAL:
                return lhs <= rhs
            case BinaryOperator.LESS_THAN:
                return lhs < rhs
            case BinaryOperator.LOGICAL_AND:
                return lhs and rhs
            case BinaryOperator.LOGICAL_OR:
                return lhs or rhs
            case BinaryOperator.MATRIX_MULTIPLY:
                return lhs @ rhs
            case BinaryOperator.MODULO:
                return lhs % rhs
            case BinaryOperator.MULTIPLY:
                return lhs * rhs
            case BinaryOperator.NOT_EQUAL:
                return lhs != rhs
            case BinaryOperator.POWER:
                return lhs ** rhs
            case BinaryOperator.RIGHT_SHIFT:
                return lhs >> rhs
            case BinaryOperator.SUBTRACT:
                return lhs - rhs
            case BinaryOperator.TRUE_DIVIDE:
                return lhs / rhs
            case _:
                raise ValueError(f'Unsupported operator: {self}')


@define_expression
class BinaryOperation(Expression):
    def __init__(self, lhs, rhs, operator: BinaryOperator):
        super().__init__()
        self.__lhs = lhs
        self.__rhs = rhs
        self.__operator = operator
        self.__hash = hash((BinaryOperation, self.__lhs, self.__rhs, self.__operator))

    def __hash__(self) -> int:
        return self.__hash

    def __bool__(self):
        match self.__operator:
            case BinaryOperator.EQUAL:
                return equal(self.__lhs, self.__rhs)
            case BinaryOperator.NOT_EQUAL:
                return not equal(self.__lhs, self.__rhs)
            case _:
                raise TypeError()

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not BinaryOperation:
            return False
        return equal(self.__lhs, other.__lhs) and equal(self.__rhs, other.__rhs) and self.__operator == other.__operator

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        lhs = self.__lhs.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__lhs) else self.__lhs
        rhs = self.__rhs.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__rhs) else self.__rhs
        if hook is not None:
            overwrite, result = hook(self, self.__operator, lhs, rhs)
            if overwrite:
                return result
        return self.__operator.apply(lhs, rhs)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__lhs):
            self.__lhs.leaves(leaves)
        if is_expression(self.__rhs):
            self.__rhs.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_left = self.__lhs.match(expression) if is_expression(self.__lhs) else False
        match_right = self.__rhs.match(expression) if is_expression(self.__rhs) else False
        return match_left or match_right

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        lhs = self.__lhs.substitute(substitution) if is_expression(self.__lhs) else self.__lhs
        rhs = self.__rhs.substitute(substitution) if is_expression(self.__rhs) else self.__rhs
        result = BinaryOperation(lhs, rhs, self.__operator)
        return substitution[result] if result in substitution else result


@define_expression
class Branch(Expression):
    def __init__(self, test: MaybeExpression, positive: MaybeExpression, negative: MaybeExpression):
        super().__init__()
        self.__test = test
        self.__positive = positive
        self.__negative = negative
        self.__hash = hash((Branch, self.__test, self.__positive, self.__negative))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Branch:
            return False
        return equal(self.__test, other.__test) and equal(self.__positive, other.__positive) and equal(self.__negative, other.__negative)

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        test = self.__test.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__test) else self.__test
        value = (self.__positive.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__positive) else self.__positive) if test \
            else (self.__negative.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__negative) else self.__negative)
        if hook is not None:
            overwrite, result = hook(self, None, value)
            if overwrite:
                return result
        return value

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__test):
            self.__test.leaves(leaves)
        if is_expression(self.__positive):
            self.__positive.leaves(leaves)
        if is_expression(self.__negative):
            self.__negative.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_test = self.__test.match(expression) if is_expression(self.__test) else False
        match_positive = self.__positive.match(expression) if is_expression(self.__positive) else False
        match_negative = self.__negative.match(expression) if is_expression(self.__negative) else False
        return match_test or match_positive or match_negative

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        test = self.__test.substitute(substitution) if is_expression(self.__test) else self.__test
        positive = self.__positive.substitute(substitution) if is_expression(self.__positive) else self.__positive
        negative = self.__negative.substitute(substitution) if is_expression(self.__negative) else self.__negative
        result = Branch(test, positive, negative)
        return substitution[result] if result in substitution else result


@define_expression
class Comprehension(Expression):
    def __init__(self, target: Expression, iterable: MaybeExpression, conditions: typing.Iterable[MaybeExpression], is_async: bool):
        super().__init__()
        self.__target = target
        self.__iterable = iterable
        self.__conditions = tuple(conditions)
        self.__is_async = is_async
        self.__hash = hash((Comprehension, self.__target, self.__iterable, self.__conditions, self.__is_async))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Comprehension:
            return False
        return equal(self.__target, other.__target) and equal(self.__iterable, other.__iterable) and self.__conditions == other.__conditions and self.__is_async == other.__is_async

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        iterable = self.__iterable.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__iterable) else self.__iterable
        if hook is not None:
            overwrite, result = hook(self, None, iterable)
            if overwrite:
                return result
        for value in iterable:
            self.__target.evaluate(local_namespace, global_namespace, hook, value)
            if all(condition.evaluate(local_namespace, global_namespace, hook) if is_expression(condition) else condition for condition in self.__conditions):
                yield

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        if is_expression(self.__iterable):
            self.__iterable.leaves(leaves)
        for condition in self.__conditions:
            if is_expression(condition):
                condition.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_iterable = self.__iterable.match(expression) if is_expression(self.__iterable) else False
        match_conditions = any((condition.match(expression) if is_expression(condition) else False for condition in self.__conditions))
        return match_target or match_iterable or match_conditions

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        iterable = self.__iterable.substitute(substitution) if is_expression(self.__iterable) else self.__iterable
        conditions = tuple((condition.substitute(substitution) if is_expression(condition) else condition for condition in self.__conditions))
        result = Comprehension(target, iterable, conditions, self.__is_async)
        return substitution[result] if result in substitution else result


@define_expression
class Contains(Expression):
    def __init__(self, target: MaybeExpression, item: MaybeExpression):
        super().__init__()
        self.__target = target
        self.__item = item
        self.__hash = hash((Contains, self.__target, self.__item))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Contains:
            return False
        return equal(self.__target, other.__target) and equal(self.__item, other.__item)

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        target = self.__target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__target) else self.__target
        item = self.__item.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__item) else self.__item
        if hook is not None:
            overwrite, result = hook(self, None, target, item)
            if overwrite:
                return result
        return item in target

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        if is_expression(self.__item):
            self.__item.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_item = self.__item.match(expression) if is_expression(self.__item) else False
        return match_target or match_item

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        item = self.__item.substitute(substitution) if is_expression(self.__item) else self.__item
        result = Contains(target, item)
        return substitution[result] if result in substitution else result


@define_expression
class Dictionary(Expression):
    def __init__(self, keys: typing.Iterable[MaybeExpression], values: typing.Iterable[MaybeExpression]):
        super().__init__()
        self.__keys = tuple(keys)
        self.__values = tuple(values)
        self.__hash = hash((Dictionary, self.__keys, self.__values))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Dictionary:
            return False
        return self.__keys == other.__keys and self.__values == other.__values

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        keys = (key.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(key) else key for key in self.__keys)
        values = (value.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(value) else value for value in self.__values)
        evaluated = {}
        for key, value in zip(keys, values):
            if key is None:
                evaluated.update(value)
            else:
                evaluated[key] = value
        if hook is not None:
            overwrite, result = hook(self, None, evaluated)
            if overwrite:
                return result
        return evaluated

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        for key in self.__keys:
            if is_expression(key):
                key.leaves(leaves)
        for value in self.__values:
            if is_expression(value):
                value.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_keys = any(key.match(expression) if is_expression(key) else False for key in self.__keys)
        match_values = any(value.match(expression) if is_expression(value) else False for value in self.__values)
        return match_keys or match_values

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        keys = (key.substitute(substitution) if is_expression(key) else key for key in self.__keys)
        values = (value.substitute(substitution) if is_expression(value) else value for value in self.__values)
        result = Dictionary(keys, values)
        return substitution[result] if result in substitution else result


@define_expression
class DictionaryComprehension(Expression):
    def __init__(self, key: MaybeExpression, value: MaybeExpression, generators: typing.Iterable[Comprehension]):
        super().__init__()
        self.__key = key
        self.__value = value
        self.__generators = tuple(generators)
        self.__hash = hash((DictionaryComprehension, self.__key, self.__value, self.__generators))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not DictionaryComprehension:
            return False
        return equal(self.__key, other.__key) and equal(self.__value, other.__value) and self.__generators == other.__generators

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        evaluated = {key: value for key, value in recursive_generator(Tuple((self.__key, self.__value), Context.Load), local_namespace, global_namespace, hook, assignment, *self.__generators)}
        if hook is not None:
            overwrite, result = hook(self, None, evaluated)
            if overwrite:
                return result
        return evaluated

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__key):
            self.__key.leaves(leaves)
        if is_expression(self.__value):
            self.__value.leaves(leaves)
        for generator in self.__generators:
            generator.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_key = self.__key.match(expression) if is_expression(self.__key) else False
        match_value = self.__value.match(expression) if is_expression(self.__value) else False
        match_generators = any((generator.match(expression) for generator in self.__generators))
        return match_key or match_value or match_generators

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        key = self.__key.substitute(substitution) if is_expression(self.__key) else self.__key
        value = self.__value.substitute(substitution) if is_expression(self.__value) else self.__value
        generators = tuple((generator.substitute(substitution) for generator in self.__generators))
        result = DictionaryComprehension(key, value, generators)
        return substitution[result] if result in substitution else result


@define_expression
class Format(Expression):
    def __init__(self, value: MaybeExpression, conversion: int, format_specification: typing.Optional[str | Expression]):
        super().__init__()
        self.__value = value
        self.__conversion = conversion
        self.__format_specification = format_specification if format_specification is not None else ''
        self.__hash = hash((Format, self.__value, self.__conversion, self.__format_specification))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Format:
            return False
        return equal(self.__value, other.__value) and self.__conversion == other.__conversion and self.__format_specification == other.__format_specification

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        value = self.__value.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__value) else self.__value
        format_specification = self.__format_specification.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__format_specification) else self.__format_specification
        if hook is not None:
            overwrite, result = hook(self, None, value, format_specification)
            if overwrite:
                return result
        match self.__conversion:
            case -1:
                return format(value, format_specification)
            case 97:
                return format(ascii(value), format_specification)
            case 114:
                return format(repr(value), format_specification)
            case 115:
                return format(str(value), format_specification)
            case _:
                raise ValueError(f"unsupported conversion: {self.__conversion}")

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__value):
            self.__value.leaves(leaves)
        if is_expression(self.__format_specification):
            self.__format_specification.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_value = self.__value.match(expression) if is_expression(self.__value) else False
        match_format_specification = self.__format_specification.match(expression) if is_expression(self.__format_specification) else False
        return match_value or match_format_specification

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        value = self.__value.substitute(substitution) if is_expression(self.__value) else self.__value
        format_specification = self.__format_specification.substitute(substitution) if is_expression(self.__format_specification) else self.__format_specification
        result = Format(value, self.__conversion, format_specification)
        return substitution[result] if result in substitution else result


@define_expression
class FunctionCall(Expression):
    def __init__(self, function: Expression | typing.Callable, args: typing.Iterable[MaybeExpression], kwargs: typing.Mapping[str, MaybeExpression]):
        super().__init__()
        self.__function = function
        self.__args = tuple(args)  # frozen arguments
        self.__kwargs = tuple(kwargs.items())  # frozen keyword arguments
        self.__hash: typing.Optional[int] = None

    def __hash__(self) -> int:
        if self.__hash is None:
            self.__hash = hash((FunctionCall, self.__function, self.__args, self.__kwargs))
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not FunctionCall:
            return False
        return equal(self.__function, other.__function) and self.__args == other.__args and self.__kwargs == other.__kwargs

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        function = self.__function.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__function) else self.__function
        arguments = []
        for item in self.__args:
            if is_expression(item):
                value = item.evaluate(local_namespace, global_namespace, hook, assignment)
                if type(item) is Starred:
                    arguments.extend(value)
                else:
                    arguments.append(value)
            else:
                arguments.append(item)
        keyword_arguments = {}
        for key, item in self.__kwargs:
            if is_expression(item):
                value = item.evaluate(local_namespace, global_namespace, hook, assignment)
                if key is None:
                    for k, v in value.items():
                        if k in keyword_arguments:
                            raise TypeError(f"multiple values for argument '{k}'")
                        keyword_arguments[k] = v
                else:
                    if key in keyword_arguments:
                        raise TypeError(f"multiple values for argument '{key}'")
                    keyword_arguments[key] = value
            else:
                keyword_arguments[key] = item
        if hook is not None:
            overwrite, result = hook(self, None, function, arguments, keyword_arguments)
            if overwrite:
                return result
        return function(*arguments, **keyword_arguments)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__function):
            self.__function.leaves(leaves)
        for item in self.__args:
            if is_expression(item):
                item.leaves(leaves)
        for _, item in self.__kwargs:
            if is_expression(item):
                item.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_function = self.__function.match(expression) if is_expression(self.__function) else False
        match_args = any((item.match(expression) if is_expression(item) else False for item in self.__args))
        match_kwargs = any((item.match(expression) if is_expression(item) else False for _, item in self.__kwargs))
        return match_function or match_args or match_kwargs

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        function = self.__function.substitute(substitution) if is_expression(self.__function) else self.__function
        args = [item.substitute(substitution) if is_expression(item) else item for item in self.__args]
        kwargs = {key: value.substitute(substitution) if is_expression(value) else value for key, value in self.__kwargs}
        result = FunctionCall(function, args, kwargs)
        return substitution[result] if result in substitution else result


@define_expression
class GeneratorComprehension(Expression):
    def __init__(self, target: MaybeExpression, generators: typing.Iterable[Comprehension]):
        super().__init__()
        self.__target = target
        self.__generators = tuple(generators)
        self.__hash = hash((GeneratorComprehension, self.__target, self.__generators))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not GeneratorComprehension:
            return False
        return equal(self.__target, other.__target) and self.__generators == other.__generators

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        generator = recursive_generator(self.__target, local_namespace, global_namespace, hook, assignment, *self.__generators)
        if hook is not None:
            overwrite, result = hook(self, None, generator)
            if overwrite:
                return result
        return generator

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        for generator in self.__generators:
            generator.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_generators = any((generator.match(expression) for generator in self.__generators))
        return match_target or match_generators

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        generators = (generator.substitute(substitution) for generator in self.__generators)
        result = GeneratorComprehension(target, generators)
        return substitution[result] if result in substitution else result


# noinspection DuplicatedCode
@define_expression
class Global(Expression):
    def __init__(self, name: str, context: Context):
        super().__init__()
        self.__name = name
        self.__context = context
        self.__hash = hash((Global, self.__name))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Global:
            return False
        return self.__name == other.__name

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        match self.__context:
            case Context.Load:
                if self.__name not in global_namespace:
                    raise NameError(f"name '{self.__name}' is not defined")
                if hook is not None:
                    overwrite, result = hook(self, None)
                    if overwrite:
                        return result
                return global_namespace[self.__name]
            case Context.Store:
                global_namespace[self.__name] = assignment
            case Context.Delete:
                del global_namespace[self.__name]

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        leaves.add(self)
        return leaves

    def match(self, expression: Expression) -> bool:
        return bool(self == expression)

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        return self


@define_expression
class Item(Expression):
    def __init__(self, target: MaybeExpression, key: MaybeExpression, context: Context):
        super().__init__()
        self.__target = target
        if type(key) is slice:
            self.__key = Slice(key.start, key.stop, key.step)
        else:
            self.__key = key
        self.__context = context
        self.__hash = hash((Item, self.__target, self.__key))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Item:
            return False
        return equal(self.__target, other.__target) and equal(self.__key, other.__key)

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        target = self.__target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__target) else self.__target
        key = self.__key.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__key) else self.__key
        match self.__context:
            case Context.Load:
                if hook is not None:
                    overwrite, result = hook(self, None, target, key)
                    if overwrite:
                        return result
                return target[key]
            case Context.Store:
                target[key] = assignment
            case Context.Delete:
                del target[key]

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        if is_expression(self.__key):
            self.__key.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_key = self.__key.match(expression) if is_expression(self.__key) else False
        return match_target or match_key

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        key = self.__key.substitute(substitution) if is_expression(self.__key) else self.__key
        result = Item(target, key, self.__context)
        return substitution[result] if result in substitution else result


@define_expression
class Join(Expression):
    def __init__(self, values: typing.Iterable[MaybeExpression]):
        super().__init__()
        self.__values = tuple(values)
        self.__hash = hash((Join, self.__values))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Join:
            return False
        return all((equal(x, y) for (x, y) in itertools.zip_longest(self.__values, other.__values)))

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        values = [value.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(value) else value for value in self.__values]
        if hook is not None:
            overwrite, result = hook(self, None, values)
            if overwrite:
                return result
        return "".join(values)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        for value in self.__values:
            if is_expression(value):
                value.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return any((item.match(expression) if is_expression(item) else False for item in self.__values))

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        values = [value.substitute(substitution) if is_expression(value) else value for value in self.__values]
        result = Join(values)
        return substitution[result] if result in substitution else result


@define_expression
class List(Expression):
    def __init__(self, items: typing.Iterable[MaybeExpression], context: Context):
        super().__init__()
        self.__items = tuple(items)
        self.__context = context
        self.__hash = hash((List, self.__items))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not List:
            return False
        return all((equal(x, y) for (x, y) in itertools.zip_longest(self.__items, other.__items)))

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        match self.__context:
            case Context.Load:
                items = []
                for item in self.__items:
                    if is_expression(item):
                        value = item.evaluate(local_namespace, global_namespace, hook, assignment)
                        if type(item) is Starred:
                            items.extend(value)
                        else:
                            items.append(value)
                    else:
                        items.append(item)
                if hook is not None:
                    overwrite, result = hook(self, None, items)
                    if overwrite:
                        return result
                return items
            case Context.Store:
                sequence_assign(self.__items, assignment, local_namespace, global_namespace, hook)
            case _:
                raise ValueError(f"unsupported context: {self.__context}")

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        for item in self.__items:
            if is_expression(item):
                item.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return any((item.match(expression) if is_expression(item) else False for item in self.__items))

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        items = [item.substitute(substitution) if is_expression(item) else item for item in self.__items]
        result = List(items, self.__context)
        return substitution[result] if result in substitution else result


@define_expression
class ListComprehension(Expression):
    def __init__(self, target: Expression, generators: typing.Iterable[Comprehension]):
        super().__init__()
        self.__target = target
        self.__generators = tuple(generators)
        self.__hash = hash((ListComprehension, self.__target, self.__generators))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not ListComprehension:
            return False
        return equal(self.__target, other.__target) and self.__generators == other.__generators

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        evaluated = [value for value in recursive_generator(self.__target, local_namespace, global_namespace, hook, assignment, *self.__generators)]
        if hook is not None:
            overwrite, result = hook(self, None, evaluated)
            if overwrite:
                return result
        return evaluated

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        for generator in self.__generators:
            generator.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_generators = any((generator.match(expression) for generator in self.__generators))
        return match_target or match_generators

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        generators = (generator.substitute(substitution) for generator in self.__generators)
        result = ListComprehension(target, generators)
        return substitution[result] if result in substitution else result


# noinspection DuplicatedCode
@define_expression
class Local(Expression):
    def __init__(self, name: str, context: Context):
        super().__init__()
        self.__name = name
        self.__context = context
        self.__hash = hash((Local, self.__name))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Local:
            return False
        return self.__name == other.__name

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        match self.__context:
            case Context.Load:
                if self.__name not in local_namespace:
                    raise NameError(f"name '{self.__name}' is not defined")
                if hook is not None:
                    overwrite, result = hook(self, None)
                    if overwrite:
                        return result
                return local_namespace[self.__name]
            case Context.Store:
                local_namespace[self.__name] = assignment
            case Context.Delete:
                del local_namespace[self.__name]

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        leaves.add(self)
        return leaves

    def match(self, expression: Expression) -> bool:
        return bool(self == expression)

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        return self


@define_expression
class Set(Expression):
    def __init__(self, items: typing.Iterable[MaybeExpression]):
        super().__init__()
        self.__items = tuple(items)
        self.__hash = hash((Set, self.__items))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Set:
            return False
        return self.__items == other.__items

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        evaluated = set(value.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(value) else value for value in self.__items)
        if hook is not None:
            overwrite, result = hook(self, None, evaluated)
            if overwrite:
                return result
        return evaluated

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        for item in self.__items:
            if is_expression(item):
                item.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return any((item.match(expression) if is_expression(item) else False for item in self.__items))

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        items = (item.substitute(substitution) if is_expression(item) else item for item in self.__items)
        result = Set(items)
        return substitution[result] if result in substitution else result


@define_expression
class SetComprehension(Expression):
    def __init__(self, target: Expression, generators: typing.Iterable[Comprehension]):
        super().__init__()
        self.__target = target
        self.__generators = tuple(generators)
        self.__hash = hash((SetComprehension, self.__target, self.__generators))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not SetComprehension:
            return False
        return equal(self.__target, other.__target) and self.__generators == other.__generators

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        evaluated = {value for value in recursive_generator(self.__target, local_namespace, global_namespace, hook, assignment, *self.__generators)}
        if hook is not None:
            overwrite, result = hook(self, None, evaluated)
            if overwrite:
                return result
        return evaluated

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        for generator in self.__generators:
            generator.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_target = self.__target.match(expression) if is_expression(self.__target) else False
        match_generators = any((generator.match(expression) for generator in self.__generators))
        return match_target or match_generators

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        generators = (generator.substitute(substitution) for generator in self.__generators)
        result = SetComprehension(target, generators)
        return substitution[result] if result in substitution else result


@define_expression
class Slice(Expression):
    def __init__(self, start: MaybeExpression, stop: MaybeExpression, step: MaybeExpression):
        super().__init__()
        self.__start = start
        self.__stop = stop
        self.__step = step
        self.__hash = hash((Slice, self.__start, self.__stop, self.__step))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Slice:
            return False
        return equal(self.__start, other.__start) and equal(self.__stop, other.__stop) and equal(self.__step, other.__step)

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        start = self.__start.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__start) else self.__start
        stop = self.__stop.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__stop) else self.__stop
        step = self.__step.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__step) else self.__step
        if hook is not None:
            overwrite, result = hook(self, None, start, stop, step)
            if overwrite:
                return result
        return slice(start, stop, step)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__start):
            self.__start.leaves(leaves)
        if is_expression(self.__stop):
            self.__stop.leaves(leaves)
        if is_expression(self.__step):
            self.__step.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        match_start = self.__start.match(expression) if is_expression(self.__start) else False
        match_stop = self.__stop.match(expression) if is_expression(self.__stop) else False
        match_step = self.__step.match(expression) if is_expression(self.__step) else False
        return match_start or match_stop or match_step

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        start = self.__start.substitute(substitution) if is_expression(self.__start) else self.__start
        stop = self.__stop.substitute(substitution) if is_expression(self.__stop) else self.__stop
        step = self.__step.substitute(substitution) if is_expression(self.__step) else self.__step
        result = Slice(start, stop, step)
        return substitution[result] if result in substitution else result


@define_expression
class Starred(Expression):
    def __init__(self, target: MaybeExpression):
        super().__init__()
        self.__target = target
        self.__hash = hash((Starred, self.__target))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Starred:
            return False
        return equal(self.__target, other.__target)

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        target = self.__target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__target) else self.__target
        if hook is not None:
            overwrite, result = hook(self, None, target)
            if overwrite:
                return result
        return target

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            self.__target.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return self.__target.match(expression) if is_expression(self.__target) else False

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        result = Starred(target)
        return substitution[result] if result in substitution else result


@define_expression
class Tuple(Expression):
    def __init__(self, items: typing.Iterable[MaybeExpression], context: Context):
        super().__init__()
        self.__items = tuple(items)
        self.__context = context
        self.__hash = hash((Tuple, self.__items))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not Tuple:
            return False
        return all((equal(x, y) for (x, y) in itertools.zip_longest(self.__items, other.__items)))

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        match self.__context:
            case Context.Load:
                items = []
                for item in self.__items:
                    if is_expression(item):
                        value = item.evaluate(local_namespace, global_namespace, hook, assignment)
                        if type(item) is Starred:
                            items.extend(value)
                        else:
                            items.append(value)
                    else:
                        items.append(item)
                items = tuple(items)
                if hook is not None:
                    overwrite, result = hook(self, None, items)
                    if overwrite:
                        return result
                return items
            case Context.Store:
                sequence_assign(self.__items, assignment, local_namespace, global_namespace, hook)
            case _:
                raise ValueError(f"unsupported context: {self.__context}")

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        for item in self.__items:
            if is_expression(item):
                item.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return any((item.match(expression) if is_expression(item) else False for item in self.__items))

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        items = [item.substitute(substitution) if is_expression(item) else item for item in self.__items]
        result = Tuple(items, self.__context)
        return substitution[result] if result in substitution else result


@enum.unique
class UnaryOperator(enum.Enum):
    INVERT = enum.auto(),
    LOGICAL_NOT = enum.auto(),
    NEGATIVE = enum.auto(),
    POSITIVE = enum.auto()

    def apply(self, target: typing.Any) -> typing.Any:
        match self:
            case UnaryOperator.INVERT:
                return ~target
            case UnaryOperator.LOGICAL_NOT:
                return not target
            case UnaryOperator.NEGATIVE:
                return -target
            case UnaryOperator.POSITIVE:
                return +target
            case _:
                raise ValueError(f"Unsupported operator: {self}")


@define_expression
class UnaryOperation(Expression):
    def __init__(self, target: MaybeExpression, operator: UnaryOperator):
        super().__init__()
        self.__target = target
        self.__operator = operator
        self.__hash = hash((UnaryOperation, self.__target, self.__operator))

    def __hash__(self) -> int:
        return self.__hash

    def equal(self, other: Expression) -> bool:
        if self is other:
            return True
        if type(other) is not UnaryOperation:
            return False
        return equal(self.__target, other.__target) and self.__operator == other.__operator

    def evaluate(self, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None, assignment: typing.Any = None) -> typing.Any:
        target = self.__target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(self.__target) else self.__target
        if hook is not None:
            overwrite, result = hook(self, self.__operator, target)
            if overwrite:
                return result
        return self.__operator.apply(target)

    def leaves(self, leaves: typing.Optional[typing.Set[Expression]] = None) -> typing.Set[Parameter | Local | Global]:
        if leaves is None:
            leaves = set()
        if is_expression(self.__target):
            return self.__target.leaves(leaves)
        return leaves

    def match(self, expression: Expression) -> bool:
        if self == expression:
            return True
        return self.__target.match(expression) if is_expression(self.__target) else False

    def substitute(self, substitution: typing.Mapping[Expression, MaybeExpression]) -> MaybeExpression:
        if self in substitution:
            return substitution[self].substitute(substitution)
        target = self.__target.substitute(substitution) if is_expression(self.__target) else self.__target
        result = UnaryOperation(target, self.__operator)
        return substitution[result] if result in substitution else result


OperatorType = typing.Optional[BinaryOperator | UnaryOperator | typing.Callable]


def sequence_assign(items: typing.Tuple, values: typing.Any, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook] = None):
    index = 0
    starred = False
    for item in items:
        if type(item) is Starred:
            if starred:
                raise ValueError("only one starred expression is allowed in assignment")
            else:
                starred = True
            item.evaluate(local_namespace, global_namespace, hook, values[index:len(values) - len(items) + index + 1])
            index += len(values) - len(items)
        else:
            if not is_expression(item):
                raise TypeError("assignment targets must be a sequence of expressions")
            item.evaluate(local_namespace, global_namespace, hook, values[index])
        index += 1


def recursive_generator(target: MaybeExpression, local_namespace: NamespaceType, global_namespace: NamespaceType, hook: typing.Optional[EvaluateHook], assignment: typing.Any, generator: Comprehension, *remaining_generators: Comprehension):
    if len(remaining_generators) <= 0:
        for _ in generator.evaluate(local_namespace, global_namespace, hook, assignment):
            yield target.evaluate(local_namespace, global_namespace, hook, assignment) if is_expression(target) else target
    else:
        for _ in generator.evaluate(local_namespace, global_namespace, hook, assignment):
            yield from recursive_generator(target, local_namespace, global_namespace, hook, assignment, *remaining_generators)


# used for function call lazy expressions where the function is an actual function while some of the arguments are lazy expressions
def call(function: typing.Callable) -> typing.Callable[..., FunctionCall]:
    def wrapper(*args: MaybeExpression, **kwargs: MaybeExpression) -> FunctionCall:
        return FunctionCall(function, args, kwargs)

    return wrapper


def logical_and(lhs: MaybeExpression, rhs: MaybeExpression) -> BinaryOperation:
    return BinaryOperation(lhs, rhs, BinaryOperator.LOGICAL_AND)


def logical_not(target: MaybeExpression) -> UnaryOperation:
    return UnaryOperation(target, UnaryOperator.LOGICAL_NOT)


def logical_or(lhs: MaybeExpression, rhs: MaybeExpression) -> BinaryOperation:
    return BinaryOperation(lhs, rhs, BinaryOperator.LOGICAL_OR)


def keyword_is(lhs: MaybeExpression, rhs: MaybeExpression) -> BinaryOperation:
    return BinaryOperation(lhs, rhs, BinaryOperator.IS)


def keyword_is_not(lhs: MaybeExpression, rhs: MaybeExpression) -> BinaryOperation:
    return BinaryOperation(lhs, rhs, BinaryOperator.IS_NOT)


def variable_name(expression: Global | Local) -> str:
    t = type(expression)
    if t is not Global and t is not Local:
        raise TypeError()
    return getattr(expression, f"_{type(expression).__name__}__name")


def retrieve_attribute(target: Expression, name: str) -> typing.Any:
    if not is_expression(target):
        raise TypeError()
    return getattr(target, f"_{type(target).__name__}__{name}")


def main():
    s = p[0]
    t = 10 + call(len)(s) + 1
    a = s[1:2]
    a = a.substitute({s: [1, 2, 3]})
    t = t.substitute({s: [1, 2, 3]})
    a.evaluate(locals(), globals())
    result = t.evaluate(locals(), globals())
    print("finished")
    pass


if __name__ == '__main__':
    main()
