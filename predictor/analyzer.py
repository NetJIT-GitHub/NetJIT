from __future__ import annotations

import ast
import inspect
import itertools
import time
import types
import typing

from predictor import base, reporter
from predictor import codes
from predictor import expressions
from predictor import history
from predictor import utility
from predictor import visitors

Rule = typing.Callable[..., None]

analyze_scope = None
max_analyze_depth: int = 1000
enable_all_hooks: bool = False
max_tracing_time: base.DurationType = float("+inf")
min_tracing_possibility: base.PossibilityType = 0.0
using_path_history: bool = False
optimize_path: bool = False
cache_path: bool = False

report: typing.Optional[reporter.Reporter] = None
min_report_time: base.DurationType = float("+inf")
max_report_time: base.DurationType = float("-inf")

excluded_codes: typing.Set[types.CodeType] = set()
analyzers: typing.MutableMapping[base.HookIndex, Analyzer] = {}
heavy_rules: typing.MutableMapping[typing.Tuple[typing.Optional[typing.Type], typing.Any], typing.Set[Rule]] = {}
light_rules: typing.MutableMapping[typing.Tuple[typing.Optional[typing.Type], typing.Any], typing.Set[Rule]] = {}
transforming = False


def define_light_rule(t: typing.Optional[typing.Type], o: typing.Any, r: Rule):
    if (t, o) not in light_rules:
        light_rules[(t, o)] = set()
    light_rules[(t, o)].add(r)


def define_heavy_rule(t: typing.Optional[typing.Type], o: typing.Any, r: Rule):
    if (t, o) not in heavy_rules:
        heavy_rules[(t, o)] = set()
    heavy_rules[(t, o)].add(r)


class TraceTarget:
    def __init__(self, target: expressions.Expression, source: base.HookIndex, possibility: base.PossibilityType, duration: base.DurationType, path: typing.Optional[utility.Path[base.HookIndex]], counter: int, tracing_depth: int):
        self.target: expressions.Expression = target
        self.source = source
        self.possibility = possibility
        self.duration = duration
        self.path = path
        self.counter: int = counter
        self.tracing_depth: int = tracing_depth

    def step(self, target: expressions.Expression, step_possibility: base.PossibilityType, step_duration: base.DurationType, last_hook: base.HookIndex, runtime: base.Runtime) -> TraceTarget:
        possibility = step_possibility * self.possibility
        duration = self.duration + step_duration
        path = runtime.step_path(last_hook, self.path, optimize_path, cache_path)
        counter = self.counter if last_hook != self.source else self.counter + 1
        return TraceTarget(target, self.source, possibility, duration, path, counter, self.tracing_depth + 1)

    def __len__(self) -> int:
        return self.tracing_depth

    def __eq__(self, other: TraceTarget) -> bool:
        if self is other:
            return True
        if type(other) is not TraceTarget:
            return False
        return self.target == other.target and self.source == other.source and self.path == other.path

    def __hash__(self) -> int:
        return hash((TraceTarget, self.target, self.source, self.path))


def should_continue_tracing(target: TraceTarget) -> bool:
    return len(target) < max_analyze_depth and target.duration < max_report_time and target.possibility > min_tracing_possibility


def transform_function(c: typing.Callable) -> typing.Callable:
    global transforming
    if transforming:
        return c
    if inspect.isbuiltin(c):
        return c
    f = codes.retrieve_function(c)
    if f is None or not hasattr(f, "__code__"):
        return c
    if f.__code__ in excluded_codes:
        return c
    if codes.check_scope(f, analyze_scope) and f.__code__ not in base.runtime_map:
        transforming = True
        node = codes.retrieve_ast(f)
        function_call_hooker = visitors.Hooker(codes.retrieve_symbol_tables(f))
        function_call_hooker.visit(node)
        hooks = function_call_hooker.hooks
        patcher = visitors.Patcher(function_call_hooker.patch)
        node = ast.Module(body=[patcher.visit(node)], type_ignores=[])
        ast.fix_missing_locations(node)
        code = compile(node, inspect.getfile(f), "exec")
        global_namespace = f.__globals__
        local_namespace = {}
        exec(code, global_namespace, local_namespace)
        modified = local_namespace[f.__name__]
        original_code = f.__code__
        code_object = codes.retrieve_code_object(modified)
        f.__code__ = code_object
        assert f.__code__ not in base.runtime_map
        metadata = base.Metadata(code_object, original_code, hooks)
        runtime = base.Runtime(metadata, max_analyze_depth)
        base.runtime_map[f.__code__] = runtime
        transforming = False
    return c


def retrieve_analyzer(index: base.HookIndex) -> Analyzer:
    if index not in analyzers:
        analyzers[index] = Analyzer(index)
    return analyzers[index]


def substitute(expression: expressions.MaybeExpression, facts: typing.Mapping[expressions.MaybeExpression, expressions.MaybeExpression]) -> expressions.MaybeExpression:
    return expression.substitute(facts) if expressions.is_expression(expression) else expression


def check_relation(target: expressions.MaybeExpression, value: expressions.MaybeExpression) -> bool:
    return (expressions.is_expression(value) and not value.match(target)) or (not expressions.is_expression(value) and target != value)


class AnalyzeEnvironment:
    class Frame:
        def __init__(self):
            self.forward_relations: typing.Set[typing.Tuple[expressions.MaybeExpression, expressions.MaybeExpression]] = set()
            self.backward_relations: typing.Set[typing.Tuple[expressions.MaybeExpression, expressions.MaybeExpression]] = set()
            self.forward_tracing: typing.Set[expressions.MaybeExpression] = set()
            self.backward_tracing: typing.Set[expressions.MaybeExpression] = set()

    def __init__(self):
        self.stack: typing.List[AnalyzeEnvironment.Frame] = []
        self.current_frame: AnalyzeEnvironment.Frame = AnalyzeEnvironment.Frame()
        self.backward_facts: typing.MutableMapping[expressions.MaybeExpression, expressions.MaybeExpression] = {}
        self.forward_facts: typing.MutableMapping[expressions.MaybeExpression, expressions.MaybeExpression] = {}
        self.backward_tracing: typing.Set[expressions.Expression] = set()
        self.forward_tracing: typing.Set[expressions.Expression] = set()
        self.modified: bool = False
        self.modified_result: typing.Any = None
        self.depth = 0

    def __bool__(self):
        return self.depth > 0

    def __enter__(self):
        if self.depth == 0:
            self.stack.clear()
            self.backward_facts.clear()
            self.forward_facts.clear()
            self.backward_tracing.clear()
            self.forward_tracing.clear()
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.append(self.current_frame)
        self.current_frame = AnalyzeEnvironment.Frame()
        self.modified = False
        self.modified_result = None
        self.depth -= 1
        if self.depth == 0:
            self.commit()

    def trace_relation(self, expression: expressions.MaybeExpression, result: expressions.MaybeExpression):
        self.trace_backward_relation(expression, result)
        self.trace_forward_relation(result, expression)

    def trace_backward_relation(self, expression: expressions.MaybeExpression, result: expressions.MaybeExpression):
        self.current_frame.backward_relations.add((expression, result))

    def trace_forward_relation(self, expression: expressions.MaybeExpression, result: expressions.MaybeExpression):
        self.current_frame.forward_relations.add((expression, result))

    def trace_backward(self, expression: expressions.MaybeExpression):
        self.current_frame.backward_tracing.add(expression)

    def trace_forward(self, expression: expressions.MaybeExpression):
        self.current_frame.forward_tracing.add(expression)

    def commit(self):
        for frame in self.stack:
            for backward_tracing in frame.backward_tracing:
                self.backward_tracing.add(substitute(backward_tracing, self.backward_facts))
            facts = {}
            for target, value in frame.backward_relations:
                target = substitute(target, self.backward_facts)
                value = substitute(value, self.backward_facts)
                if check_relation(target, value):
                    facts[target] = value
            self.backward_facts.update(facts)
        for frame in reversed(self.stack):
            for forward_tracing in frame.forward_tracing:
                self.forward_tracing.add(substitute(forward_tracing, self.forward_facts))
            facts = {}
            for target, value in frame.forward_relations:
                target = substitute(target, self.forward_facts)
                value = substitute(value, self.forward_facts)
                if check_relation(target, value):
                    facts[target] = value
            self.forward_facts.update(facts)

    def modify(self, result: expressions.MaybeExpression):
        self.modified = True
        self.modified_result = result

    def result(self) -> typing.Tuple[bool, typing.Any]:
        return self.modified, self.modified_result


class Analyzer:
    def __init__(self, index: base.HookIndex):
        self.index = index
        self.enabled = False
        self.history: history.History = history.History()
        self.environment: AnalyzeEnvironment = AnalyzeEnvironment()
        self.current_frame: typing.Optional[types.FrameType] = None
        self.backward_trace_targets: typing.Set[TraceTarget] = set()
        self.forward_trace_targets: typing.Set[expressions.MaybeExpression] = set()

    def __enter__(self) -> expressions.EvaluateHook:
        assert self.current_frame is not None
        self.environment.__enter__()
        for forward_trace_target in self.forward_trace_targets:
            self.environment.trace_forward(forward_trace_target)

        def wrapper(expression: expressions.Expression, operator: typing.Any, *args, **kwargs) -> typing.Tuple[bool, typing.Any]:
            with self.environment:
                if type(expression) is expressions.FunctionCall and len(args) > 0:
                    function = utility.retrieve_positional_or_keyword_argument(0, "function", None)(*args, **kwargs)
                    operator = codes.retrieve_code_object(function)
                if self.is_enabled():
                    if operator is not None:
                        for rule in heavy_rules.get((type(expression), operator), []):
                            rule(self.environment, expression, operator, *args, **kwargs)
                    for rule in heavy_rules.get((type(expression), None), []):
                        rule(self.environment, expression, operator, *args, **kwargs)
                    for rule in heavy_rules.get((None, None), []):
                        rule(self.environment, expression, operator, *args, **kwargs)
                if operator is not None:
                    for rule in light_rules.get((type(expression), operator), []):
                        rule(self.environment, expression, operator, *args, **kwargs)
                for rule in light_rules.get((type(expression), None), []):
                    rule(self.environment, expression, operator, *args, **kwargs)
                for rule in light_rules.get((None, None), []):
                    rule(self.environment, expression, operator, *args, **kwargs)
                return self.environment.result()

        return wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        hook_info = base.hooks[self.index]
        current_code = self.current_frame.f_code
        runtime = base.runtime_map[current_code]
        for expression, result in hook_info.modifications.items():
            self.environment.trace_relation(expression, result)
        self.environment.__exit__(exc_type, exc_value, traceback)
        self.history.record(self.environment.backward_facts, self.environment.backward_tracing, self.environment.forward_tracing, runtime, using_path_history, optimize_path, cache_path)
        self.analyze_backward_tracing(runtime)
        self.analyze_forward_tracing(runtime)
        self.current_frame = None

    def __call__(self, current_frame: types.FrameType) -> Analyzer:
        self.current_frame = current_frame
        return self

    def enable(self):
        self.enabled = True

    def is_enabled(self) -> bool:
        return enable_all_hooks or self.enabled

    def analyze_backward_tracing(self, runtime: base.Runtime):
        for expression, possibility in self.history.lookup_backward_tracing().items():
            target = TraceTarget(expression, self.index, possibility, 0.0, path=runtime.step_path(self.index, None, optimize_path, cache_path), counter=1, tracing_depth=0)
            self.update_backward_trace_target(target)
        for backward_trace_target in self.backward_trace_targets:
            self.report(backward_trace_target, runtime.current_time)
            if not should_continue_tracing(backward_trace_target):
                continue
            substituted = self.history.lookup_relations(backward_trace_target.target, self.history.affected)
            for e, p in substituted.items():
                if p <= 0.0:
                    continue
                for h, i in self.history.paths.items():
                    analyzer = retrieve_analyzer(h)
                    target = backward_trace_target.step(e, p * i.counter / self.history.total, i.average_duration, h, runtime)
                    analyzer.update_backward_trace_target(target)

    def analyze_forward_tracing(self, runtime: base.Runtime):
        hook_map = runtime.metadata.hook_map
        for forward_tracing_target in self.history.lookup_forward_tracing():
            if not expressions.is_expression(forward_tracing_target):
                continue
            if forward_tracing_target in self.forward_trace_targets:
                continue
            for leaf in forward_tracing_target.leaves():
                symbol_table = codes.retrieve_symbol_table(runtime.metadata.original_code)
                symbol = symbol_table[expressions.retrieve_attribute(leaf, "name")]
                for node in itertools.chain(symbol.loads, symbol.stores, symbol.deletes):
                    if node not in hook_map:
                        continue
                    for hook_id in runtime.metadata.hook_map[node]:
                        analyzer = retrieve_analyzer(hook_id)
                        analyzer.update_forward_trace_target(forward_tracing_target)

    def update_backward_trace_target(self, target: TraceTarget):
        if target in self.backward_trace_targets:
            self.backward_trace_targets.remove(target)
        self.backward_trace_targets.add(target)
        self.enable()

    def update_forward_trace_target(self, target: expressions.MaybeExpression):
        self.forward_trace_targets.add(target)
        self.enable()

    def report(self, target: TraceTarget, current_time: base.DurationType):
        if min_report_time <= target.duration <= max_report_time and report is not None:
            local_namespace = self.current_frame.f_locals
            global_namespace = self.current_frame.f_globals
            target_hook = base.hooks[target.source]
            target_analyzer = retrieve_analyzer(target.source)
            path = target.path
            value = target.target.evaluate(local_namespace, global_namespace)
            duration = target.duration
            hit_path_cache = False
            if path in target_analyzer.history.path_history:
                duration = target_analyzer.history.path_history[path]
                hit_path_cache = True
            report(f"reporting at {self.index}:\n")
            report(f"current time:\t{time.monotonic()}\n")
            report(f"tracing target:\t{target.source}, {target_hook.counter + target.counter}\n")
            report(f"current counter:\t{target_hook.counter}\n")
            report(f"tracing value:\t{value}\n")
            report(f"path:\t{path}\n")
            report(f"tracing depth:\t{len(target)}\n")
            report(f"duration:\t{duration}\n")
            report(f"hit path cache:\t{hit_path_cache}\n")
            report(f"expected time:\t{current_time + duration}\n")
            report(f"possibility:\t{target.possibility}\n")
            report("\n")


# noinspection PyUnusedLocal
def hook(*args, **kwargs) -> typing.Any:
    def wrapper(index: base.HookIndex):
        frame = inspect.currentframe().f_back
        assert frame is not None
        local_namespace = frame.f_locals
        global_namespace = frame.f_globals
        info = base.hooks[index]
        if transforming:
            return info.expression.evaluate(local_namespace, global_namespace)
        analyzer = retrieve_analyzer(index)
        assert frame.f_code in base.runtime_map
        runtime = base.runtime_map[frame.f_code]
        runtime.record(index, time.monotonic())
        with analyzer(frame) as analyzer:
            result = info.expression.evaluate(local_namespace, global_namespace, analyzer)
        if info.is_exit:
            runtime.reset()
        info.counter += 1
        return result

    return wrapper
