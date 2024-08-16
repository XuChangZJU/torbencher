import importlib
import os
import platform
import time

import pandas as pd
import psutil

from .singleTester import SingleTester
from .testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from .util.apitools import *


class torbencherc:
    SUPPORTED_FORMATS = ["csv", 'json', 'xlsx']
    AVAILABLE_TEST_MODULES = ["torch", "torch.nn", "torch.nn.functional"]
    SUPPORTED_NAME_SPECS = ["timestamp", "datetime"]

    class DEFAULTS:
        OUT_DIR = "./results"
        FORMAT = "csv"
        TEST_MODULES = ["torch", "torch.nn", "torch.nn.functional"]
        NAME_SPEC = "timestamp"

    class ConfigKey:
        OUT_DIR = "out_dir"
        SEED = "seed"
        DEVICES = "devices"
        TEST_MODULES = "test_modules"
        FORMAT = "format"
        NUM_EPOCH = "num_epoch"
        NAME_SPEC = "name_spec"
        DEBUG = "debug"

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
        TEST_RESULTS = "test_results"
        START_TIME = "start_time"

    class TestResultKey:
        MODULE_NAME = "module_name"
        TESTCASE = "testcase"
        PASSED = "passed"
        FAILURE_DETAILS = "failure_details"
        STATUS = "status"
        COST_TIME = "cost_time(ms)"

    def __init__(self, config: dict):
        """
        **description**
        Initialize the Torbencher class with a configuration dictionary.

        **params**
        - config (dict): Configuration dictionary.

        **returns**
        - None
        """
        self.config = self.parseConfig(config)
        self.result = {}
        self.initBasics()
        self.tester = SingleTester()

    def parseConfig(self, config: dict) -> dict:
        """
        **description**
        Parse the configuration dictionary and set default values where necessary.

        **params**
        - config (dict): Configuration dictionary.

        **returns**
        - dict: Parsed configuration dictionary.
        """
        if torbencherc.ConfigKey.OUT_DIR not in config:
            print("No output directory found in config, check your result at pwd/cwd.")
            config[torbencherc.ConfigKey.OUT_DIR] = torbencherc.DEFAULTS.OUT_DIR

        if torbencherc.ConfigKey.SEED in config:
            seed = config[torbencherc.ConfigKey.SEED]
            assert isinstance(seed, int)
        else:
            config[torbencherc.ConfigKey.SEED] = time.time_ns()

        if torbencherc.ConfigKey.DEVICES in config:
            if isinstance(config[torbencherc.ConfigKey.DEVICES], list):
                pass
            elif isinstance(config[torbencherc.ConfigKey.DEVICES], str):
                config[torbencherc.ConfigKey.DEVICES] = [config[torbencherc.ConfigKey.DEVICES]]
        else:
            config[torbencherc.ConfigKey.DEVICES] = ["cpu"]

        if torbencherc.ConfigKey.TEST_MODULES in config:
            test_modules = config[torbencherc.ConfigKey.TEST_MODULES]
            assert isinstance(test_modules, list)
        else:
            config[torbencherc.ConfigKey.TEST_MODULES] = torbencherc.DEFAULTS.TEST_MODULES

        if not torbencherc.ConfigKey.FORMAT in config:
            config[torbencherc.ConfigKey.FORMAT] = torbencherc.DEFAULTS.FORMAT
        else:
            if config[torbencherc.ConfigKey.FORMAT] not in torbencherc.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format {config[torbencherc.ConfigKey.FORMAT]}. Supported formats are {torbencherc.SUPPORTED_FORMATS}")

        if not torbencherc.ConfigKey.NAME_SPEC in config:
            config[torbencherc.ConfigKey.NAME_SPEC] = torbencherc.DEFAULTS.NAME_SPEC
        else:
            if config[torbencherc.ConfigKey.NAME_SPEC] not in torbencherc.SUPPORTED_NAME_SPECS:
                raise ValueError(
                    f"Unsupported name spec {config[torbencherc.ConfigKey.NAME_SPEC]}. Supported formats are {torbencherc.SUPPORTED_NAME_SPECS}")

        if not torbencherc.ConfigKey.DEBUG in config:
            config[torbencherc.ConfigKey.DEBUG] = True
        return config

    def initBasics(self):
        """
        **description**
        Initialize basic information like start time, machine info, and software info.

        **params**
        - None

        **returns**
        - None
        """
        self.result[torbencherc.ResultKey.START_TIME] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.result[torbencherc.ResultKey.CPU] = platform.processor()
        self.result[torbencherc.ResultKey.MEMORY] = "{:.2f} GB".format(round(psutil.virtual_memory().total / (1024 ** 3), 2))

        self.result[torbencherc.ResultKey.OS] = platform.system()
        self.result[torbencherc.ResultKey.OS_RELEASE] = platform.release()
        self.result[torbencherc.ResultKey.OS_VERSION] = platform.version()
        self.result[torbencherc.ResultKey.NODE] = platform.node()
        self.result[torbencherc.ResultKey.MACHINE] = platform.machine()

        self.result[torbencherc.ResultKey.PYTHON_VERSION] = platform.python_version()
        self.result[torbencherc.ResultKey.TORCH_VERSION] = torch.__version__

    def run(self) -> dict:
        """
        **description**
        Run all the test cases on all the devices and save the results.

        **params**
        - None

        **returns**
        - dict: Test results.
        """
        testResult = self.runTest(self.config)
        self.result[torbencherc.ResultKey.TEST_RESULTS] = testResult
        self.saveResult(self.config, self.result)
        print(f"Torbencher has finished testing, check your result at {self.config[torbencherc.ConfigKey.OUT_DIR]}")
        self.deleteNonPyFiles()
        return testResult

    def runTest(self, config: dict) -> dict:
        """
        **description**
        Run the specified test cases based on the configuration.

        **params**
        - config (dict): Configuration dictionary.

        **returns**
        - dict: Test results.
        """
        if torch.__version__ < "2.0.0":
            raise RuntimeError("Torch version must be greater than 2.0.0")

        outputResults = {}
        names = [f"src.testcase.{test_module}" for test_module in config[torbencherc.ConfigKey.TEST_MODULES]]

        moduleList = self.importModules(names, outputResults)
        allTestCases = self.getTestCases(moduleList)

        outputResults = self.runWithTester(config=self.config, allTestCases=allTestCases, outputResults=outputResults)
        return outputResults

    def importModules(self, names: list, outputResults: dict) -> dict:
        """
        **description**
        Import the specified test modules.

        **params**
        - names (list): List of module names to import.
        - outputResults (dict): Dictionary to store the import results.

        **returns**
        - dict: Dictionary of imported modules.
        """
        modules = {}
        for name in names:
            try:
                module = importlib.import_module(name)
                if name not in modules:
                    modules[name] = []
                modules[name].append(module)
            except Exception as e:
                print(f"Error importing module {name}: {e}")
                outputResults[name] = {
                    torbencherc.TestResultKey.PASSED: "ModuleImportError",
                    torbencherc.TestResultKey.FAILURE_DETAILS: str(e)
                }
        return modules

    def getTestCases(self, moduleList: dict) -> dict:
        """
        **description**
        Get test cases from the imported modules.

        **params**
        - moduleList (dict): Dictionary of imported modules.

        **returns**
        - dict: Dictionary of test cases.
        """
        allTestCases = {}
        for name, testcaseModules in moduleList.items():
            allTestCases[name] = []
            for module in testcaseModules:
                attrNames = getAttributes(module)
                for attrName in attrNames:
                    attr = getattr(module, attrName, None)
                    if isinstance(attr, type) \
                            and issubclass(attr, TorBencherTestCaseBase) \
                            and attr is not TorBencherTestCaseBase:
                        allTestCases[name].append(attr)
        return allTestCases

    def runWithTester(self, config: dict, allTestCases: dict, outputResults: dict) -> dict:
        """
        **description**
        Run the test cases with the tester.

        **params**
        - config (dict): Configuration dictionary.
        - allTestCases (dict): Dictionary of test cases.
        - outputResults (dict): Dictionary to store the test results.

        **returns**
        - dict: Updated test results.
        """
        devices = config[torbencherc.ConfigKey.DEVICES]
        seed = config[torbencherc.ConfigKey.SEED]
        repeat = config[torbencherc.ConfigKey.NUM_EPOCH]
        debug = config[torbencherc.ConfigKey.DEBUG]

        def repeat_test(testCase):
            testcaseName = testCase.__name__
            outputResults[device][testModuleName][testcaseName] = {
                torbencherc.TestResultKey.STATUS: "Passed",
                torbencherc.TestResultKey.COST_TIME: []
            }
            for _ in range(repeat):
                try:
                    startTime = time.perf_counter_ns()
                    passed = self.tester.run(testCase, device=device, seed=seed, debug=debug)
                    endTime = time.perf_counter_ns()
                    costTime = (endTime - startTime) / (1000 ** 3)
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME].append(costTime)
                except Exception as e:
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.STATUS] = "CompareError"
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME] = "N/A"
                    break

                if passed == -2:
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.STATUS] = "Skipped"
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME] = "N/A"
                    break

                if passed == -1:
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.STATUS] = "Error"
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME] = "N/A"
                    break
                if not passed:
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.STATUS] = "Failed"
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME] = sum(
                        outputResults[device][testModuleName][testcaseName][
                            torbencherc.TestResultKey.COST_TIME]) / len(
                        outputResults[device][testModuleName][testcaseName][
                            torbencherc.TestResultKey.COST_TIME])
                    break
            if outputResults[device][testModuleName][testcaseName][
                torbencherc.TestResultKey.STATUS] == "Passed":
                outputResults[device][testModuleName][testcaseName][torbencherc.TestResultKey.COST_TIME] = sum(
                    outputResults[device][testModuleName][testcaseName][
                        torbencherc.TestResultKey.COST_TIME]) / len(
                    outputResults[device][testModuleName][testcaseName][torbencherc.TestResultKey.COST_TIME])

        for device in devices:
            outputResults[device] = {}
            for testModuleName, testCases in allTestCases.items():
                outputResults[device][testModuleName] = {}
                for testCase in testCases:
                    repeat_test(testCase)
            self.tester.resetTester()
        return outputResults

    def saveResult(self, config: dict, result: dict):
        """
        **description**
        Save the test results based on the specified format.

        **params**
        - config (dict): Configuration dictionary.
        - result (dict): Test results.

        **returns**
        - None
        """
        testResult = result[torbencherc.ResultKey.TEST_RESULTS]
        formattedResult = self.getDFFormattedTestResult(testResult)
        self.saveDFFormattedResult(config, formattedResult)

    def getDFFormattedTestResult(self, testResult: dict) -> pd.DataFrame:
        """
        **description**
        Format the test results into a DataFrame.

        **params**
        - testResult (dict): Dictionary of test results.

        **returns**
        - pd.DataFrame: Formatted test results as DataFrame.
        """
        rows = []
        devices = list(testResult.keys())
        header = [torbencherc.TestResultKey.MODULE_NAME, torbencherc.TestResultKey.TESTCASE]
        for device in devices:
            header.append(f"{device.upper()}_status")
            header.append(f"{device.upper()}_cost_time(ms)")

        # Use a set to track which test cases have already been processed
        processedCases = set()

        for device, testModules in testResult.items():
            for moduleName, testCases in testModules.items():
                for testCase, result in testCases.items():
                    if (moduleName, testCase) not in processedCases:
                        row = [moduleName, testCase]
                        for dev in devices:
                            status = testResult[dev].get(moduleName, {}).get(testCase, {}).get(
                                torbencherc.TestResultKey.STATUS, "N/A")
                            costTime = testResult[dev].get(moduleName, {}).get(testCase, {}).get(
                                torbencherc.TestResultKey.COST_TIME, "N/A")
                            row.append(status)
                            row.append(costTime)
                        rows.append(row)
                        # Mark this test case as processed
                        processedCases.add((moduleName, testCase))

        return pd.DataFrame(rows, columns=header)

    def getFileName(self, config: dict) -> str:
        """
        **description**
        Generate a file name based on the name specification in the configuration.

        **params**
        - config (dict): Configuration dictionary.

        **returns**
        - str: Generated file name.
        """
        if config[torbencherc.ConfigKey.NAME_SPEC] == "timestamp":
            return str(time.time_ns())
        elif config[torbencherc.ConfigKey.NAME_SPEC] == "datetime":
            return time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return "unknown_name_spec"

    def saveDFFormattedResult(self, config: dict, formattedResult: pd.DataFrame):
        """
        **description**
        Save the DataFrame formatted result based on the specified format.

        **params**
        - config (dict): Configuration dictionary.
        - formattedResult (pd.DataFrame): DataFrame of formatted results.

        **returns**
        - None
        """
        out_dir = config[torbencherc.ConfigKey.OUT_DIR]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        format = config[torbencherc.ConfigKey.FORMAT]
        fileNameSpec = self.getFileName(config)

        if format == "csv":
            formattedResult.to_csv(os.path.join(out_dir, f'torbencherc_test_result_{fileNameSpec}.csv'),
                                   index=False)
        elif format == "json":
            formattedResult.to_json(os.path.join(out_dir, f'torbencherc_test_result_{fileNameSpec}.json'),
                                    orient='records')
        elif format == "xlsx":
            formattedResult.to_excel(os.path.join(out_dir, f'torbencherc_test_result_{fileNameSpec}.xlsx'),
                                     index=False)

    def deleteNonPyFiles(self, dirPath: str = None):
        """
        **description**
        Delete non-Python files from the specified directory.

        **params**
        - dirPath (str, optional): Directory path to clean. Defaults to the current working directory.

        **returns**
        - None
        """
        if not dirPath:
            dirPath = os.path.join(os.getcwd(), "")
        for root, dirs, files in os.walk(dirPath):
            for file in files:
                if file.endswith('.cpp') or file.endswith('.pyc') or file.endswith('.c') or file.endswith('.pt'):
                    filePath = os.path.join(root, file)
                    os.remove(filePath)
