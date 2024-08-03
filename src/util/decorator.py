import functools
import time
import inspect

from ..testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from .apitools import *


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

    def class_has_method(cls, func_name):
        for name in dir(cls):
            if name.endswith(func_name):  # _Cls__method
                return True
        return False

    assert class_has_method(cls, func_name)

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


def test_api(api):  # Useless, 已弃用
    """
    **description**
    Decorator to wrap a test class so that its test methods record their results upon execution.

    **params**
    api: The operation function or method to be tested.

    **returns**
    class_decorator: A decorator to decorate a class.
    """

    @functools.wraps(api)
    def class_decorator(cls):
        """
        **description**
        Class decorator to wrap the class, ensure it meets requirements, and decorate its test methods.

        **params**
        cls: The class to be decorated.

        **returns**
        cls: The decorated class.
        """
        assert issubclass(cls, TorBencherTestCaseBase)
        # cls._api = api  # Assign the provided api to the class's _api attribute
        # for name, method in cls.__dict__.items():
        #     if callable(method) and name.startswith("test"):
        #         setattr(
        #             cls, name, decorate_test_function(cls, method, api)
        #         )  # Replace the original method with the decorated method
        return cls

    # def decorate_test_function(cls, method, operator):
    #     """
    #     **description**
    #     Method decorator to wrap test methods and record results after execution.
    #
    #     **params**
    #     cls: The class to which the method belongs.
    #     method: The method to be decorated.
    #     operator: The operation function or method being tested.
    #
    #     **returns**
    #     wrapper: The decorated method.
    #     """
    #
    #     @functools.wraps(method)
    #     def wrapper(*args, **kwargs):
    #         result = method(*args, **kwargs)
    #         return result  # Ensure the original method's result is returned
    #
    #     return wrapper

    return class_decorator


def randomInjector(func, storage, testcaseName):
    """
    Decorator that injects random values into a function and stores the results in a storage dictionary.

    Params:
    func (callable): The random function to be decorated.
    storage (dict): The storage dictionary to store results.
    testcaseName (str): The name of the test case.

    Returns:
    wrapper (function): The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        funcName = getAPIName(func)
        if testcaseName not in storage:
            storage[testcaseName] = {"result": {}, "status": False, "count": {}}
        if funcName not in storage[testcaseName]["result"]:
            storage[testcaseName]["result"][funcName] = []
        if funcName not in storage[testcaseName]["count"]:
            storage[testcaseName]["count"][funcName] = 0

        if not storage[testcaseName]["status"]:
            rst = func(*args, **kwargs)
            storage[testcaseName]["result"][funcName].append(rst)
            return rst
        else:
            result = storage[testcaseName]["result"][funcName][storage[testcaseName]["count"][funcName]]
            storage[testcaseName]["count"][funcName] += 1
            if storage[testcaseName]["count"][funcName] == len(storage[testcaseName]["result"][funcName]):
                storage[testcaseName]["count"][funcName] = 0
            return result

    return wrapper
