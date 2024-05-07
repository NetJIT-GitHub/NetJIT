from __future__ import annotations

import ast
import inspect
import typing

from predictor import expressions

T = typing.TypeVar("T")


class Path(typing.Generic[T]):
    def __init__(self, current: T, remaining: typing.Optional[Path[T]]):
        self.current = current
        self.remaining = remaining
        self.length = 1 + (len(remaining) if remaining is not None else 0)
        self.hash = hash((Path, self.current, self.remaining))
        self.representation: typing.Optional[str] = None

    def __contains__(self, item: T):
        if self.current == item:
            return True
        if self.remaining is None:
            return False
        return item in self.remaining

    def __iter__(self):
        yield self.current
        if self.remaining is not None:
            yield from self.remaining

    def __len__(self):
        return self.length

    def __repr__(self) -> str:
        if self.representation is None:
            self.representation = repr(self.current) if self.remaining is None else f"{repr(self.current)}, {repr(self.remaining)}"
        return self.representation

    def __eq__(self, other: Path) -> bool:
        if self is other:
            return True
        if type(other) is not Path:
            return False
        if self.length != other.length:
            return False
        return self.current == other.current and self.remaining == other.remaining

    def __hash__(self) -> int:
        return self.hash

    def __add__(self, other: Path[T]) -> Path[T]:
        if self.remaining is None:
            return Path(self.current, other)
        if other is None:
            return self
        return Path(self.current, self.remaining + other)

    def __radd__(self, other: Path[T]) -> Path[T]:
        if other is None:
            return self
        return other + self


class RingQueue(typing.Generic[T]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: typing.List[typing.Optional[T]] = [None] * capacity
        self.start = 0
        self.end = 0
        self.length = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

    def __reversed__(self):
        for i in range(self.length):
            yield self[-1 - i]

    def __getitem__(self, item: int):
        offset = item % self.length
        while offset < 0:
            offset += self.length
        return self.storage[(self.start + offset) % self.capacity]

    def is_empty(self) -> bool:
        return self.length <= 0

    def is_full(self) -> bool:
        return self.length >= self.capacity

    def clear(self):
        while not self.is_empty():
            self.dequeue()

    def enqueue(self, data: T):
        if self.is_full():
            self.dequeue()
        self.storage[self.end] = data
        self.end += 1
        self.length += 1
        if self.end >= self.capacity:
            self.end -= self.capacity

    def dequeue(self) -> T:
        if self.is_empty():
            raise IndexError("Queue is empty")
        data = self.storage[self.start]
        self.storage[self.start] = None
        self.start += 1
        self.length -= 1
        if self.start >= self.capacity:
            self.start -= self.capacity
        return data


def smooth_average(factor: float, previous: T, current: T) -> T:
    return previous * factor + current * (1 - factor)


def retrieve_positional_argument(index: int, default=None):
    def wrapper(*args, **kwargs):
        if index < len(args):
            return args[index]
        return default


def retrieve_keyword_argument(name: str, default=None):
    def wrapper(*args, **kwargs):
        return kwargs.get(name, default)

    return wrapper


def retrieve_positional_or_keyword_argument(index: int, name: str, default=None):
    def wrapper(*args, **kwargs):
        if index < len(args):
            return args[index]
        return kwargs.get(name, default)

    return wrapper


def retrieve_argument(f: typing.Callable, name: str):
    signature = inspect.signature(f)

    def wrapper(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        return arguments.arguments[name]

    return wrapper


def convert_context(ctx: ast.expr_context) -> expressions.Context:
    match ctx:
        case ast.Load():
            return expressions.Context.Load
        case ast.Store():
            return expressions.Context.Store
        case ast.Del():
            return expressions.Context.Delete
        case _:
            raise ValueError(f"Invalid context {ctx}")
