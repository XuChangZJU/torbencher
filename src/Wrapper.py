import functools
import json
import os
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
    DEFAULT_FORMAT = "csv"
    SUPPORTED_FORMATS = ["json", "csv"]
    SUPPORETD_NAME_SPEC = ["timestamp", "datetime", "serial"]

    class ConfigKey:
        OUT_DIR = "out_dir"
        FORMAT = "format"
        FILE_MAX_SIZE = "file_max_size"
        FILE_NAME_SPEC = "file_name_spec"

    class ResultKey:
        COUNT = "count"
        TOTAL_TIME = "total_time(ms)"
        SCALE = "scale"

        class ScaleKey:
            CALL_NUMBER = "call#"
            START_TIMESTAMP = "start_timestamp"
            COST_TIME = "cost_time(ms)"

    def __init__(self, config: dict):
        self.call_count = {}  # 创建一个字典来存储调用信息
        self.config = self._parse_config(config)

    def _parse_config(self, config: dict):
        if Wrapper.ConfigKey.OUT_DIR not in config:
            raise ValueError("Output directory is required")

        if Wrapper.ConfigKey.FORMAT in config:
            assert isinstance(config[Wrapper.ConfigKey.FORMAT], str)
            format = config[Wrapper.ConfigKey.FORMAT]
            if format not in Wrapper.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format {format} for saving result")
        else:
            config[Wrapper.ConfigKey.FORMAT] = Wrapper.DEFAULT_FORMAT

        if Wrapper.ConfigKey.FILE_MAX_SIZE in config:
            assert isinstance(config[Wrapper.ConfigKey.FILE_MAX_SIZE], str)

        if Wrapper.ConfigKey.FILE_NAME_SPEC in config:
            assert isinstance(config[Wrapper.ConfigKey.FILE_NAME_SPEC], str)
            name_spec = config[Wrapper.ConfigKey.FILE_NAME_SPEC]
            if name_spec not in Wrapper.SUPPORETD_NAME_SPEC:
                raise ValueError(f"Unsupported file name spec {name_spec}")

        return config

    def start(self, func, *args, **kwargs):
        self.decorate_module(torch)
        ret = func(*args, **kwargs)
        self.save_result(
            self.config[Wrapper.ConfigKey.OUT_DIR],
            self.config[Wrapper.ConfigKey.FORMAT],
            self.config[Wrapper.ConfigKey.FILE_MAX_SIZE],
            self.config[Wrapper.ConfigKey.FILE_NAME_SPEC],
        )
        return ret

    def save_result(
        self, path: str, format: str, file_max_size: str, file_name_spec: str
    ):
        def parse_file_max_size_str(file_max_size: str):
            max_size = 0
            if file_max_size.endswith("KB"):
                max_size = int(file_max_size[:-2]) * 1024
            elif file_max_size.endswith("MB"):
                max_size = int(file_max_size[:-2]) * 1024 * 1024
            elif file_max_size.endswith("GB"):
                max_size = int(file_max_size[:-2]) * 1024 * 1024 * 1024
            else:
                raise ValueError(f"Unsupported file max size format {file_max_size}")

            if max_size <= 0:
                raise ValueError("File max size must be positive")

            return max_size

        def get_file_name_suffix(file_name_spec: str):
            if file_name_spec == "timestamp":
                return time.time_ns()
            elif file_name_spec == "datetime":
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            elif file_name_spec == "serial":
                raise NotImplementedError

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError(f"Path {path} is not a directory")
        else:
            os.makedirs(path)

        max_size = parse_file_max_size_str(file_max_size)
        name_suffix = get_file_name_suffix(file_name_spec)
        if format == "json":
            self.save_json_result(path, max_size, name_suffix)
        elif format == "csv":
            self.save_csv_result(path, max_size, name_suffix)
        else:
            raise NotImplementedError

    def save_json_result(self, path: str, max_size: int, name_suffix: str):
        file_name = f"wrapper_result_{name_suffix}.json"
        with open(file_name, "w") as f:
            json.dump(self.call_count, f)

    def save_csv_result(self, path: str, max_size: int, name_suffix: str):
        file_name = f"wrapper_result_{name_suffix}.csv"
        with open(os.path.join(path, file_name), "w") as f:
            f.write("api_name,call_number,start_time,cost_time(ms),scale\n")
            for api_name, api_info in self.call_count.items():
                scale_list = api_info[Wrapper.ResultKey.SCALE]
                for scale_obj in scale_list:
                    assert isinstance(scale_obj, dict)

                    call_number = scale_obj[Wrapper.ResultKey.ScaleKey.CALL_NUMBER]
                    start_time = scale_obj[Wrapper.ResultKey.ScaleKey.START_TIMESTAMP]
                    cost_time = scale_obj[Wrapper.ResultKey.ScaleKey.COST_TIME]

                    scale_obj.pop(Wrapper.ResultKey.ScaleKey.CALL_NUMBER)
                    scale_obj.pop(Wrapper.ResultKey.ScaleKey.START_TIMESTAMP)
                    scale_obj.pop(Wrapper.ResultKey.ScaleKey.COST_TIME)

                    scale_str = json.dumps(scale_obj, ensure_ascii=False)
                    f.write(
                        f"{api_name},{call_number},{start_time},{cost_time},{scale_str}\n"
                    )

    def call_count_decorator(self, func, module_name=None):
        """author: zym, fg"""

        def get_scale_info(args: tuple):
            """
            获取参数的维度信息
            """
            scale_info = []
            for i in range(len(args)):
                arg = args[i]
                if type(arg) == torch.Tensor:
                    scale_info.append((f"arg{i}_size", arg.size()))  # (desc, value)
            return scale_info

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time_ns()
            result, elapsed_time = timing_decorator(func)(
                *args, **kwargs
            )  # 调用函数并计时

            full_name = module_name + "." + func.__name__
            if full_name not in self.call_count:
                self.call_count[full_name] = {
                    Wrapper.ResultKey.COUNT: 0,
                    Wrapper.ResultKey.TOTAL_TIME: 0.0,
                    Wrapper.ResultKey.SCALE: [],
                }

            log_str = "call #{:d} of {:s}, start at {}, cost {:f} ms".format(
                self.call_count[full_name][Wrapper.ResultKey.COUNT],
                full_name,
                start_time,
                elapsed_time,
            )

            scale_obj = {}
            scale_obj[Wrapper.ResultKey.ScaleKey.CALL_NUMBER] = self.call_count[
                full_name
            ][Wrapper.ResultKey.COUNT]
            scale_obj[Wrapper.ResultKey.ScaleKey.START_TIMESTAMP] = start_time
            scale_obj[Wrapper.ResultKey.ScaleKey.COST_TIME] = elapsed_time

            scale_info = get_scale_info(args)
            for desc, value in scale_info:
                scale_obj[desc] = value

            self.call_count[full_name][Wrapper.ResultKey.COUNT] += 1
            self.call_count[full_name][Wrapper.ResultKey.TOTAL_TIME] += elapsed_time
            self.call_count[full_name][Wrapper.ResultKey.SCALE].append(scale_obj)

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
