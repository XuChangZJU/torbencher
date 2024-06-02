import platform
import time
import psutil
import torch


class torbencher:
    SUPPORTED_FORMATS = ["json"]
    AVAILABLE_TEST_MODULES = ["torch", "torch.nn", "torch.nn.functional"]

    class ConfigKey:
        SEED = "seed"
        DEVICES = "devices"
        TEST_MODULES = "test_modules"
        FORMAT = "format"

    class ResultKey:
        CPU = "cpu"
        MEMORY = "memory"
        OS = "os"
        OS_RELEASE = "os_release"
        OS_VERSION = "os_version"
        NODE = "node"
        MACHINE = "machine"
        PYTHON_VERSION = "python_version"
        TORCH_VERSION = "torch_version"
        RESULTS = "results"
        START_TIME = "start_time"

    def __init__(self, config: dict):
        self.config = self._parse_config(config)

    def _parse_config(self, config: dict):
        if torbencher.ConfigKey.SEED in config:
            seed = config[torbencher.ConfigKey.SEED]
            assert isinstance(seed, int)
        else:
            config[torbencher.ConfigKey.SEED] = time.time_ns()

        if torbencher.ConfigKey.DEVICES in config:
            devices = config[torbencher.ConfigKey.DEVICES]
            assert isinstance(devices, list)
            if "cpu" not in devices:
                config[torbencher.ConfigKey.DEVICES].insert(0, "cpu")
        else:
            config[torbencher.ConfigKey.DEVICES] = ["cpu"]

        if torbencher.ConfigKey.TEST_MODULES in config:
            test_modules = config[torbencher.ConfigKey.TEST_MODULES]
            assert isinstance(test_modules, list)
        else:
            config[torbencher.ConfigKey.TEST_MODULES] = (
                torbencher.AVAILABLE_TEST_MODULES
            )

        if torbencher.ConfigKey.FORMAT in config:
            format = config[torbencher.ConfigKey.FORMAT]
            assert isinstance(format, str)
            if format not in torbencher.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format {format}. Supported formats are {torbencher.SUPPORTED_FORMATS}"
                )

        return config

    def run(self):
        return self._get_test_result(self.config[torbencher.ConfigKey.FORMAT])

    def _get_test_result(self, format: str):
        if format == "json":
            return self._get_json_test_result()
        else:
            raise NotImplementedError

    def _get_json_test_result(self):
        result = {}

        # start time
        result[torbencher.ResultKey.START_TIME] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )

        # machine info
        result[torbencher.ResultKey.CPU] = platform.processor()
        result[torbencher.ResultKey.MEMORY] = "{:.2f} GB".format(
            round(psutil.virtual_memory().total / (1024**3), 2)
        )

        # os info
        result[torbencher.ResultKey.OS] = platform.system()
        result[torbencher.ResultKey.OS_RELEASE] = platform.release()
        result[torbencher.ResultKey.OS_VERSION] = platform.version()
        result[torbencher.ResultKey.NODE] = platform.node()
        result[torbencher.ResultKey.MACHINE] = platform.machine()

        # software info
        result[torbencher.ResultKey.PYTHON_VERSION] = platform.python_version()
        result[torbencher.ResultKey.TORCH_VERSION] = torch.__version__

        # test results
        seed = self.config[torbencher.ConfigKey.SEED]
        devices = self.config[torbencher.ConfigKey.DEVICES]
        test_modules = self.config[torbencher.ConfigKey.TEST_MODULES]
        result[torbencher.ResultKey.RESULTS] = self._run_tests(
            seed, devices, test_modules
        )

        return result

    def _run_tests(self, seed: int, devices: list, test_modules: list):
        test_results = []
        return test_results
