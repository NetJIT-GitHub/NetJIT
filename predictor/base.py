from __future__ import annotations

import ast
import types
import typing

from predictor import expressions, utility, codes, cfg

HookIndex = int
Counter = int
ModificationsType = typing.Mapping[expressions.Expression, expressions.MaybeExpression]
RelationsType = typing.Mapping[expressions.Expression, expressions.MaybeExpression]
BackwardTracingType = typing.Set[expressions.Expression]
ForwardTracingType = typing.Set[expressions.Expression]
DurationType = float
PossibilityType = float

hooks: typing.MutableMapping[HookIndex, HookInfo] = {}
paths: typing.MutableMapping[typing.Tuple[HookIndex, typing.Optional[utility.Path[HookIndex]]], utility.Path[HookIndex]] = {}
runtime_map: typing.MutableMapping[types.CodeType, Runtime] = {}


class Metadata:
    def __init__(self, code: types.CodeType, original_code: types.CodeType, hook_map: typing.Mapping[ast.AST, typing.Set[HookIndex]]):
        self.code = code
        self.original_code = original_code
        self.hook_map = hook_map


class Runtime:
    def __init__(self, metadata: Metadata, hook_history_size: int):
        self.metadata = metadata
        self.hook_history: utility.RingQueue[typing.Tuple[HookIndex, DurationType]] = utility.RingQueue(hook_history_size)
        self.last_hook: HookIndex = -1
        self.current_hook: HookIndex = -1
        self.last_time: DurationType = 0
        self.current_time: DurationType = 0

    def build_path(self, hook_index: HookIndex, path: typing.Optional[utility.Path[HookIndex]], optimize: bool) -> utility.Path[HookIndex]:
        if path is None or path.remaining is None:
            return utility.Path(hook_index, path)
        if optimize:
            current_cfg = codes.retrieve_cfg(self.metadata.code)
            start = hooks[hook_index].statement
            end = hooks[path.current].statement
            if current_cfg.is_sequential(start, end) == cfg.Sequential.ALWAYS:
                return utility.Path(hook_index, path.remaining)
            else:
                return utility.Path(hook_index, path)
        else:
            return utility.Path(hook_index, path)

    def step_path(self, hook_index: HookIndex, path: typing.Optional[utility.Path[HookIndex]], optimize: bool, cache: bool) -> utility.Path[HookIndex]:
        if cache:
            index = (hook_index, path)
            if index not in paths:
                paths[index] = self.build_path(hook_index, path, optimize)
            return paths[index]
        else:
            return self.build_path(hook_index, path, optimize)

    def record(self, current_hook: HookIndex, current_time: DurationType):
        self.hook_history.enqueue((current_hook, current_time))
        self.last_hook = self.current_hook
        self.last_time = self.current_time
        self.current_hook = current_hook
        self.current_time = current_time

    def reset(self):
        self.hook_history.clear()
        self.last_hook = -1
        self.current_hook = -1
        self.last_time = 0
        self.current_time = 0


class HookInfo:
    def __init__(self, index: HookIndex, expression: expressions.Expression, modifications: ModificationsType, statement: ast.AST, is_exit: bool = False, is_yield: bool = False):
        self.index = index
        self.counter: Counter = 0
        self.expression = expression
        self.modifications = modifications
        self.statement = statement
        self.is_exit = is_exit
        self.is_yield = is_yield
