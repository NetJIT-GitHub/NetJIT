from __future__ import annotations

import typing

from predictor import base, utility
from predictor import expressions

SmoothFactor = 0.875


class HistoryRecord:
    def __init__(self, relations: base.RelationsType, backward_tracing: base.BackwardTracingType, forward_tracing: base.ForwardTracingType):
        self.relations = {**relations}
        self.backward_tracing = frozenset(backward_tracing)
        self.forward_tracing = frozenset(forward_tracing)
        self.hash = hash((HistoryRecord, tuple(relations.items()), self.backward_tracing, self.forward_tracing))

    def __eq__(self, other: HistoryRecord) -> bool:
        return self.relations == other.relations and self.backward_tracing == other.backward_tracing

    def __hash__(self) -> int:
        return self.hash


class PathInfo:
    def __init__(self):
        self.counter = 0
        self.average_duration: base.DurationType = 0  # in seconds

    def update(self, duration: base.DurationType):
        self.counter += 1
        self.average_duration = utility.smooth_average(SmoothFactor, self.average_duration, duration)


class CacheIndex:
    def __init__(self, expression: expressions.Expression, eliminate: typing.Set[expressions.Expression]):
        self.expression = expression
        self.eliminate = frozenset(eliminate)

    def __hash__(self) -> int:
        return hash((CacheIndex, self.expression, self.eliminate))

    def __eq__(self, other) -> bool:
        if type(other) is not CacheIndex:
            return False
        if self.expression != other.expression or self.eliminate != other.eliminate:
            return False
        return True


T = typing.TypeVar("T")


class CacheEntry(typing.Generic[T]):
    def __init__(self, matched: typing.Set[T], index: int):
        self.matched = matched
        self.index = index


RelationCacheType = typing.Tuple[HistoryRecord, expressions.MaybeExpression]


class History:
    def __init__(self):
        self.total = 0
        self.affected: typing.Set[expressions.Expression] = set()
        self.paths: typing.MutableMapping[base.HookIndex, PathInfo] = {}
        self.path_history: typing.MutableMapping[utility.Path[base.HookIndex], base.DurationType] = {}
        self.relation_cache: typing.MutableMapping[CacheIndex, CacheEntry[RelationCacheType]] = {}
        self.backward_tracing_cache: CacheEntry[HistoryRecord] = CacheEntry(set(), 0)
        self.forward_tracing_cache: CacheEntry[HistoryRecord] = CacheEntry(set(), 0)
        self.storage: typing.MutableMapping[HistoryRecord, int] = {}
        self.records: typing.List[HistoryRecord] = []

    def record(self, relations: base.RelationsType, backward_tracing: base.BackwardTracingType, forward_tracing: base.ForwardTracingType, runtime: base.Runtime, using_path_history: bool, optimize_path: bool, cache_path: bool):
        for e in relations:
            self.affected.add(e)
        record = HistoryRecord(relations, backward_tracing, forward_tracing)
        if record not in self.storage:
            self.storage[record] = 0
            self.records.append(record)
        self.storage[record] += 1
        self.total += 1
        if len(backward_tracing) > 0 and using_path_history:
            self.update_path_history(runtime, optimize_path, cache_path)
        if runtime.last_hook < 0:
            return
        if runtime.last_hook not in self.paths:
            self.paths[runtime.last_hook] = PathInfo()
        self.paths[runtime.last_hook].update(runtime.current_time - runtime.last_time)

    def update_path_history(self, runtime: base.Runtime, optimize_path: bool, cache_path: bool):
        path: typing.Optional[utility.Path[base.HookIndex]] = None
        for index, timestamp in reversed(runtime.hook_history):
            path = runtime.step_path(index, path, optimize_path, cache_path)
            if path in self.path_history:
                self.path_history[path] = utility.smooth_average(SmoothFactor, self.path_history[path], runtime.current_time - timestamp)
            else:
                self.path_history[path] = runtime.current_time - timestamp

    def lookup_relations(self, expression: expressions.Expression, eliminate: typing.Set[expressions.Expression]) -> typing.Mapping[expressions.MaybeExpression, base.PossibilityType]:
        index = CacheIndex(expression, eliminate)
        if index not in self.relation_cache:
            self.relation_cache[index] = CacheEntry(set(), 0)
        entry = self.relation_cache[index]
        result: typing.MutableMapping[expressions.MaybeExpression, float] = {}
        for record, substituted in entry.matched:
            result[substituted] = result.get(substituted, 0.0) + self.storage[record] / self.total
        while entry.index < len(self.records):
            record = self.records[entry.index]
            entry.index += 1
            substituted = expression.substitute(record.relations)
            if not expressions.is_expression(substituted):
                entry.matched.add((record, substituted))
                result[substituted] = result.get(substituted, 0.0) + self.storage[record] / self.total
                continue
            if not any((substituted.match(e) for e in eliminate)):
                entry.matched.add((record, substituted))
                result[substituted] = result.get(substituted, 0.0) + self.storage[record] / self.total
                continue
        return result

    def lookup_backward_tracing(self) -> typing.Mapping[expressions.MaybeExpression, base.PossibilityType]:
        result: typing.MutableMapping[expressions.MaybeExpression, float] = {}
        while self.backward_tracing_cache.index < len(self.records):
            record = self.records[self.backward_tracing_cache.index]
            self.backward_tracing_cache.index += 1
            if len(record.backward_tracing) > 0:
                self.backward_tracing_cache.matched.add(record)
        for record in self.backward_tracing_cache.matched:
            for tracing in record.backward_tracing:
                result[tracing] = result.get(tracing, 0.0) + 1.0 * self.storage[record] / self.total
        return result

    def lookup_forward_tracing(self) -> typing.Iterable[expressions.MaybeExpression]:
        result: typing.Set[expressions.MaybeExpression] = set()
        while self.forward_tracing_cache.index < len(self.records):
            record = self.records[self.forward_tracing_cache.index]
            self.forward_tracing_cache.index += 1
            if len(record.forward_tracing) > 0:
                result.update(record.forward_tracing)
        return result
