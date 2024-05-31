import functools
import json
import time
import types

import torch


def timing_decorator(func):
    """author: fg"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        ret = func(*args, **kwargs)
        end = time.perf_counter_ns()
        delta = (end - start) / 1000 / 1000  # ns -> us -> ms
        timing_log = f"{func.__name__}() cost {delta} ms"
        return ret, delta

    return wrapper


class Wrapper:
    def __init__(self, config: dict):
        self.config = config
        self.call_count = {}  # 创建一个字典来存储调用信息

    def start(self, func, *args, **kwargs):
        self.decorate_module(torch)
        ret = func(*args, **kwargs)
        self.save_result()
        return ret

    def save_result(self):
        """
        save call_count to json file
        :return: call_count object
        """
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        with open(f"call_count_{timestamp}.json", "w") as f:
            json.dump(self.call_count, f)

    def call_count_decorator(self, func, module_name=None):
        """author: zym, fg"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            COUNT = "count"
            TOTAL_TIME = "total_time(ms)"
            SCALE = "scale"
            CALL_NUMBER = "call#"

            result, elapsed_time = timing_decorator(func)(
                *args, **kwargs
            )  # 调用函数并计时

            full_name = module_name + "." + func.__name__
            if full_name not in self.call_count:
                self.call_count[full_name] = {COUNT: 0, TOTAL_TIME: 0.0, SCALE: []}

            log_str = "call #{:d} of {:s}, cost {:f} ms".format(
                self.call_count[full_name][COUNT], full_name, elapsed_time
            )
            scale_obj = {}  # Tensor维度信息
            for i in range(len(args)):  # 分析参数表
                arg = args[i]
                if type(arg) == torch.Tensor:
                    if CALL_NUMBER not in scale_obj:
                        scale_obj[CALL_NUMBER] = self.call_count[full_name][COUNT]
                    scale_obj[f"arg{i}_size"] = arg.size()

            self.call_count[full_name][COUNT] += 1
            self.call_count[full_name][TOTAL_TIME] += elapsed_time
            if CALL_NUMBER in scale_obj:
                self.call_count[full_name][SCALE].append(scale_obj)

            return result

        wrapper._is_decorated = True
        return wrapper

    def set_new_attr(self, module, attr_name, attr):
        """author: zym"""
        if not hasattr(attr, "_is_decorated"):
            decorated_attr = self.call_count_decorator(attr, module.__name__)
            decorated_attr._is_decorated = True
            setattr(module, attr_name, decorated_attr)

    def decorate_module(self, module, visited=None):
        """
        递归封装所有的包

        author: zym
        """
        module_name = module.__name__

        if visited is None:
            visited = set()
        if module_name in visited:
            return
        visited.add(module_name)

        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, types.FunctionType):
                    self.set_new_attr(module, attr_name, attr)
                    # print(f"Decorated function: {module_name}.{attr_name}")
                elif isinstance(attr, types.ModuleType) and attr.__name__.startswith(
                    "torch"
                ):
                    # print(f"Descending into module: {attr.__name__}")
                    self.decorate_module(attr, visited)
                elif isinstance(attr, type):
                    # print(f"Descending into class: {attr.__name__} in {module_name}")
                    self.decorate_class(attr)
                elif callable(attr):
                    self.set_new_attr(module, attr_name, attr)
            except AttributeError:
                continue

    def decorate_class(self, cls):
        """author: zym"""
        for attr_name in dir(cls):
            # if attr_name.startswith('__') and attr_name.endswith('__'):
            #     continue  # Skip special attributes
            try:
                attr = getattr(cls, attr_name)
                if isinstance(attr, types.FunctionType):
                    self.set_new_attr(cls, attr_name, attr)
                elif attr_name in [
                    "__add__",
                    "__mul__",
                    "__sub__",
                    "__truediv__",
                    "__matmul__",
                    "__pow__",
                    "__mod__",
                ]:
                    self.set_new_attr(cls, attr_name, attr)  # 特殊处理运算符重载方法
            except (AttributeError, TypeError) as e:
                continue
