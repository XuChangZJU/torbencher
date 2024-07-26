import functools
import time
import inspect

from ..testcase.TorBencherTestCaseBase import TorBencherTestCaseBase


def get_class_that_defined_method(meth):
    """
    https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3
    """
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
            inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__,
                                                                                                '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def is_static_method_of_class(func, cls=None):
    if cls is None:
        result = get_class_that_defined_method(func)
        if result is None:
            return False
        cls = result

    assert inspect.isclass(cls)

    func_name = func.__name__
    # if func_name == "<lambda>":
    #     return False

    # debug
    # with open("test.log" , "a") as f:
    #     f.write(f"{cls}\t{func.__qualname__}\t\n")

    # def class_has_method(cls, func_name):
    #     for name in dir(cls):
    #         if name.endswith(func_name): # _Cls__method
    #             return True
    #     return False
    # assert class_has_method(cls, func_name)

    return isinstance(inspect.getattr_static(cls, func_name), staticmethod)


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        ret = func(*args, **kwargs)  # original function, e.g. torch.add
        end = time.perf_counter_ns()
        delta = (end - start) / 1000 / 1000  # ns -> us -> ms
        timing_log = f"{func.__name__}() cost {delta} ms"
        return ret, delta

    if is_static_method_of_class(func) == True:  # func is a static method
        return staticmethod(wrapper)
    else:
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
            result = method(*args, **kwargs)
            assert hasattr(cls, "final_stats")
            stat_obj = {
                "operator_name": f"{operator.__qualname__}",
                "operator_result": result,
                "specs": None,
            }
            cls.final_stats.append(stat_obj)
            return result;

        return wrapper

    return class_decorator
