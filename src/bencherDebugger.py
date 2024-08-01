import importlib
import inspect
import time
import unittest
import torch
import csv
import logging

from .testcase.TorBencherTestCaseBase import TorBencherTestCaseBase

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class bencherDebugger:
    SUPPORTED_FORMATS = ["json"]
    AVAILABLE_TEST_MODULES = ["torch", "torch.nn", "torch.nn.functional"]

    class ConfigKey:
        SEED = "seed"
        DEVICES = "devices"
        TEST_MODULES = "test_modules"
        FORMAT = "format"
        INCLUDING_SUCCESS = "including_success"
        NUM_EPOCHES = "num_epoches"
        RESULT_FILE = "result_file"

    class ResultKey:
        TESTCASE = "testcase"
        STATUS = "status"
        # ERROR = "error"
        ERRORS = "errors"
        ERROR_DETAILS = "error_details"
        FAILURES = "failures"
        FAILURE_DETAILS = "failure_details"

    def __init__(self, config: dict):
        """
        Initialize the benchdebugger with the given configuration.

        **params**
        - config (dict): The configuration dictionary.

        **returns**
        - None
        """
        self.config = self._parse_config(config)

    def _parse_config(self, config: dict) -> dict:
        """
        Parse and validate the configuration dictionary.

        **params**
        - config (dict): The configuration dictionary.

        **returns**
        - dict: The parsed configuration dictionary.
        """
        config.setdefault(bencherDebugger.ConfigKey.SEED, time.time_ns())
        config.setdefault(bencherDebugger.ConfigKey.DEVICES, ["cpu"])
        config.setdefault(bencherDebugger.ConfigKey.TEST_MODULES, bencherDebugger.AVAILABLE_TEST_MODULES)
        config.setdefault(bencherDebugger.ConfigKey.INCLUDING_SUCCESS, False)
        config.setdefault(bencherDebugger.ConfigKey.NUM_EPOCHES, 1)
        config.setdefault(bencherDebugger.ConfigKey.RESULT_FILE, "test_results.csv")

        if bencherDebugger.ConfigKey.FORMAT not in config or config[
            bencherDebugger.ConfigKey.FORMAT] not in bencherDebugger.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format. Supported formats are {bencherDebugger.SUPPORTED_FORMATS}")

        return config

    def run(self) -> None:
        """
        Run the tests based on the configuration.

        **params**
        - None

        **returns**
        - None
        """
        self.runTest(self.config)

    def runTest(self, config: dict) -> None:
        """
        Execute the test cases based on the configuration.

        **params**
        - config (dict): The configuration dictionary.

        **returns**
        - None
        """
        outputResults = {}
        names = [f"src.testcase.{test_module}" for test_module in config[bencherDebugger.ConfigKey.TEST_MODULES]]
        modules = self.importModules(names, outputResults)

        if torch.__version__ < "2.1.0":
            raise RuntimeError("Torch version must be greater than 2.1.0")

        testcases_info = self._discoverTestcases(modules, outputResults)

        torch.manual_seed(config[bencherDebugger.ConfigKey.SEED])
        for _ in range(config[bencherDebugger.ConfigKey.NUM_EPOCHES]):
            for device in config[bencherDebugger.ConfigKey.DEVICES]:
                logging.debug(f"Running tests on device: {device}")
                for module_info in testcases_info:
                    for testcase in module_info["testcases"]:
                        name, result = self.runTestcase(testcase)
                        if config[bencherDebugger.ConfigKey.INCLUDING_SUCCESS]:
                            if name not in outputResults:
                                outputResults[name] = (result)
                            elif result["status"] != "Success":
                                outputResults[name] = (result)
                            else:
                                pass
                        else:
                            if result["status"] == "Failed":
                                outputResults[name] = (result)
                            else:
                                pass

        try:
            self._write_results_to_csv(outputResults)
        except Exception as e:
            logging.error(f"Error writing results to CSV: {e}")

    def importModules(self, names: list, outputResults: dict) -> list:
        """
        Import the specified test modules.

        **params**
        - names (list): List of module names to import.
        - output_results (list): List to store the import results.

        **returns**
        - list: List of imported modules.
        """
        modules = []
        for name in names:
            try:
                module = importlib.import_module(name)
                modules.append(module)
            except Exception as e:
                logging.error(f"Error importing module {name}: {e}")
                outputResults[name] = {

                    bencherDebugger.ResultKey.STATUS: "ModuleImportError",
                    bencherDebugger.ResultKey.ERROR_DETAILS: str(e)
                }
        return modules

    def _discoverTestcases(self, modules: list, outputResults: dict) -> list:
        """
        Discover test cases from the imported modules.

        **params**
        - modules (list): List of imported modules.
        - output_results (list): List to store the discovery results.

        **returns**
        - list: List of discovered test cases information.
        """

        def discoverTestcases(module):
            assert inspect.ismodule(module), "The provided input is not a module"

            testcases = [attr for name, attr in inspect.getmembers(module, inspect.isclass) if
                         issubclass(attr, TorBencherTestCaseBase) and attr is not TorBencherTestCaseBase]
            return testcases

        testcases_info = []
        for module in modules:
            try:
                module_info = {
                    "module": module,
                    "testcases": discoverTestcases(module),
                }
                testcases_info.append(module_info)
                # logging.debug(f"Got testcases in module {module}: {e}")
            except Exception as e:
                logging.error(f"Error discovering testcases in module {module}: {e}")
                outputResults[str(module)] = {
                    bencherDebugger.ResultKey.STATUS: "Error",
                    "error": str(e)
                }
        return testcases_info

    def runTestcase(self, testcase: unittest.TestCase) -> dict:
        """
        Runs the given unittest testcase and logs the results.

        **params**
        - testcase (unittest.TestCase): The unittest testcase class to run.

        **returns**
        - dict: Dictionary with test results.
        """
        loader = unittest.TestLoader()
        runner = unittest.TextTestRunner(verbosity=2)

        suite = loader.loadTestsFromTestCase(testcase)

        try:
            result = runner.run(suite)

            error_details = [{"test": test.id(), "error": err} for test, err in result.errors]
            failure_details = [{"test": test.id(), "failure": fail} for test, fail in result.failures]

            result_summary = {
                bencherDebugger.ResultKey.STATUS: "Success" if result.wasSuccessful() else "Failed",
                bencherDebugger.ResultKey.ERRORS: len(result.errors),
                bencherDebugger.ResultKey.FAILURES: len(result.failures),
                bencherDebugger.ResultKey.ERROR_DETAILS: error_details,
                bencherDebugger.ResultKey.FAILURE_DETAILS: failure_details
            }

            for error in error_details:
                logging.error("Error: %s\n%s", error["test"], error["error"])

            for failure in failure_details:
                logging.error("Failure: %s\n%s", failure["test"], failure["failure"])

            return testcase.__name__, result_summary
        except Exception as e:
            result_summary = {
                bencherDebugger.ResultKey.STATUS: "UnitTestError",
                bencherDebugger.ResultKey.ERRORS: 1,
                bencherDebugger.ResultKey.FAILURES: 0,
                bencherDebugger.ResultKey.ERROR_DETAILS: [str(e)],
                bencherDebugger.ResultKey.FAILURE_DETAILS: []
            }
            logging.error("UnitTestError: %s", e)
            return testcase.__name__, result_summary

    def _write_results_to_csv(self, results: dict) -> None:
        """
        Write the test results to a CSV file.

        **params**
        - results (dict): Dictionary of test results, where the key is the test case name and the value is a dictionary of test result data.

        **returns**
        - None
        """
        if not results:
            logging.error("No results to write to CSV")
            return
        formattedResults = []

        # Collect all possible fieldnames dynamically
        # fieldnames = set(["testcase"])
        for name, result in results.items():
            result["testcase"] = name
            formattedResults.append(result)
            # fieldnames.update(result.keys())
        fieldnames = [getattr(bencherDebugger.ResultKey, col) for col in dir(bencherDebugger.ResultKey) if
                      "__" not in col]

        with open(self.config.get(bencherDebugger.ConfigKey.RESULT_FILE), 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=list(fieldnames))
            dict_writer.writeheader()
            dict_writer.writerows(formattedResults)
