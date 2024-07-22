# torbencher.py
import importlib
import inspect
import platform
	@@ -6,10 +7,8 @@
import psutil
import torch
import torch.version
from .testcase.TorBencherTestCaseBase import TorBencherTestCaseBase;
from .util.unitest import MyTestRunner, MyTestLoader;

class torbencher:
    SUPPORTED_FORMATS = ["json"]
	@@ -56,17 +55,13 @@ def _parse_config(self, config: dict):
            test_modules = config[torbencher.ConfigKey.TEST_MODULES]
            assert isinstance(test_modules, list)
        else:
            config[torbencher.ConfigKey.TEST_MODULES] = torbencher.AVAILABLE_TEST_MODULES

        if torbencher.ConfigKey.FORMAT in config:
            format = config[torbencher.ConfigKey.FORMAT]
            assert isinstance(format, str)
            if format not in torbencher.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format {format}. Supported formats are {torbencher.SUPPORTED_FORMATS}")

        return config

	@@ -83,15 +78,11 @@ def _get_json_test_result(self):
        result = {}

        # start time
        result[torbencher.ResultKey.START_TIME] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # machine info
        result[torbencher.ResultKey.CPU] = platform.processor()
        result[torbencher.ResultKey.MEMORY] = "{:.2f} GB".format(round(psutil.virtual_memory().total / (1024**3), 2))

        # os info
        result[torbencher.ResultKey.OS] = platform.system()
	@@ -108,17 +99,15 @@ def _get_json_test_result(self):
        seed = self.config[torbencher.ConfigKey.SEED]
        devices = self.config[torbencher.ConfigKey.DEVICES]
        test_modules = self.config[torbencher.ConfigKey.TEST_MODULES]
        result[torbencher.ResultKey.RESULTS] = self._run_tests(seed, devices, test_modules)

        return result

    def _run_tests(self, seed: int, devices: list, test_modules: list):
        output_results = {}

        loader = MyTestLoader();
        runner = MyTestRunner(verbosity=2);

        names = [f"src.testcase.{test_module}" for test_module in test_modules]
        modules = [importlib.import_module(name) for name in names]
	@@ -139,18 +128,19 @@ def discover_testcases(module):
                "testcases": discover_testcases(module),
            }
            testcases_info.append(module_info)

        if torch.__version__ < "2.0.0":
            raise RuntimeError("Torch version must be greater than 2.0.0")

        torch.manual_seed(seed)
        for device in devices:
            # torch.set_default_device(device);
            output_results[device] = [];
            for module_info in testcases_info:
                for testcase in module_info["testcases"]:
                    suite = loader.loadTestsFromTestCase(testcase)
                    result = runner.run(suite)
                    output_results[device].append(result.getReturnValues());


        return output_results
