from __future__ import annotations

import functools
import os
import time
import typing

from predictor import analyzer, base, codes, reporter


def entrypoint(analyze_scope=os.getcwd(), max_analyze_depth: int = 100,
               max_tracing_time: base.DurationType = 15.0, min_tracing_possibility: base.PossibilityType = 0.01,
               using_path_history: bool = True, optimize_path: bool = True, cache_path: bool = True,
               min_report_time: base.DurationType = 0.01, max_report_time: base.DurationType = 10.0, report_file="./report.txt"):
    def decorator(function: typing.Callable) -> typing.Callable:
        function = analyzer.transform_function(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with reporter.FileReporter(report_file) as report:
                report("clock synchronization:\n")
                report(f"system clock:\t{time.time()}\n")
                report(f"monotonic clock:\t{time.monotonic()}\n")
                report("\n")
                analyzer.report = report
                function(*args, **kwargs)
            analyzer.report = None

        return wrapper

    analyzer.analyze_scope = analyze_scope
    analyzer.max_analyze_depth = max_analyze_depth
    analyzer.max_tracing_time = max_tracing_time
    analyzer.min_tracing_possibility = min_tracing_possibility
    analyzer.using_path_history = using_path_history
    analyzer.optimize_path = optimize_path
    analyzer.cache_path = cache_path

    analyzer.min_report_time = min_report_time
    analyzer.max_report_time = max_report_time

    return decorator


def exclude(*callables: typing.Callable):
    def decorator(function: typing.Callable) -> typing.Callable:
        return function

    for c in callables:
        code = codes.retrieve_code_object(c)
        if code is not None:
            analyzer.excluded_codes.add(code)
    return decorator
