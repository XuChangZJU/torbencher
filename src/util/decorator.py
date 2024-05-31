import functools


from src.testcase.TorBencherBase import TorBencherTestCaseBase


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
