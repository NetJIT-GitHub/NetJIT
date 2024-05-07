from __future__ import annotations

import inspect
import types
import typing

from .. import expressions, analyzer


def recursive_function_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: types.CodeType, function: typing.Any, *args, **kwargs):
    analyzer.transform_function(function)


def register_rules():
    analyzer.define_light_rule(t=expressions.FunctionCall, o=None, r=recursive_function_hook)


def enable_common_rules(c: typing.Callable) -> typing.Callable:
    register_rules()
    return c


if __name__ == 'common':
    register_rules()
