import functools
import time

from ..testcase.TorBencherTestCaseBase import TorBencherTestCaseBase


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        ret = func(*args, **kwargs)
        end = time.perf_counter_ns()
        delta = (end - start) / 1000 / 1000  # ns -> us -> ms
        timing_log = f"{func.__name__}() cost {delta} ms"
        return ret, delta

    return wrapper


def test_api(api):
    def class_decorator(cls):
        assert issubclass(cls, TorBencherTestCaseBase)
        cls._api = api
        cls.final_stats = []  # statistics list
        for name, method in cls.__dict__.items():
            if callable(method) and name.startswith("test"):
                setattr(
                    cls, name, decorate_test_function(cls, method, api)
                )  # 用装饰后的方法替换原始方法
        return cls

    def decorate_test_function(cls, method, operator):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            result, specs = method(*args, **kwargs)
            assert hasattr(cls, "final_stats")
            stat_obj = {
                "operator_name": f"{operator.__module__}.{operator.__name__}",
                "operator_result": result,
                "specs": specs,
            }
            cls.final_stats.append(stat_obj)

        return wrapper

    return class_decorator
