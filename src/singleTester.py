import random
import numpy as np
import time
import torch
import unittest

from .testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from .util.apitools import *
from .util.decorator import randomInjector
from .util.CustomUnittest import MyTestRunner, MyTestLoader


class SingleTester:
    """
    **description**
    SingleTester is a class responsible for running test cases derived from TorBencherTestCaseBase on both CPU and optionally a specified device (like GPU) while ensuring consistent random seeds.

    **Attributes**
    loader (MyTestLoader): Instance for loading test cases.
    runner (MyTestRunner): Instance for running test suites.
    storage (dict): Dictionary to store intermediate values for random number injections.

    **Methods**
    run(testcase: TorBencherTestCaseBase, device: str = None, seed: int = 443, debug: bool = True) -> bool or int:
        Runs the provided test case on CPU and optionally on a specified device, comparing the results.
    applyRandomInjectors(testcaseName: str) -> None:
        Apply the randomInjector decorator to random functions.
    sendValueToDevice(testcaseName: str, storage: dict, device: str) -> None:
        Send the values from storage to the specified device.
    """

    def __init__(self) -> None:
        """
        **description**
        Initializes the SingleTester class by setting up the test loader, runner, and storage.

        **params**
        None
        """
        self.loader = MyTestLoader()
        self.runner = MyTestRunner(verbosity=2)
        self.uloader = unittest.TestLoader()
        self.urunner = unittest.TextTestRunner()
        self.storage = {}
        self.uniform = random.uniform
        self.rrandint = random.randint
        self.trandint = torch.randint
        self.randn = torch.randn
        self.normal = torch.normal
        self.rand = torch.rand
        self.rchoice = random.choice
        self.rrandom = random.random
        self.rshuffle = random.shuffle
        self.rrandperm = torch.randperm

    def run(self, testcase: TorBencherTestCaseBase, device: str = "cpu", seed: int = None,
            debug: bool = True) -> bool or int:
        """
        **description**
        Runs the provided test case on CPU and optionally on a specified device, comparing the results.

        **params**
        testcase (TorBencherTestCaseBase): The test case class to be tested.
        device (str, optional): The device to test on (e.g., 'cuda'). Defaults to cpu.
        seed (int, optional): The seed for random number generation. Defaults to 443.
        debug (bool, optional): If True, enables debug print statements. Defaults to True.

        **returns**
        passed (bool): The passed status of the testcase.

        **raises**
        AssertionError: If the provided testcase is not a subclass of TorBencherTestCaseBase.
        Warning/Error: If one of CPU or Device has no return.
        """
        assert issubclass(testcase, TorBencherTestCaseBase)
        testcaseName = testcase.__name__
        if seed is None:
            seed = time.time_ns()

        # Pre Check for whether to skipped
        if debug:
            print(f"Start precheck for {testcaseName}")
        suite = self.uloader.loadTestsFromTestCase(testcase)
        preResult = self.urunner.run(suite)
        if bool(preResult.skipped):
            print(f"[SKIPPED] {testcaseName} skipped, return -2")
            return -2

        if device != "cpu":
            print(f"[INITIALIZE] Start testing {testcaseName} on {device}")
        else:
            print(f"[INITIALIZE] Start testing {testcaseName}")

        suite = self.loader.loadTestsFromTestCase(testcase)
        self.sendValueToDevice(testcaseName, self.storage, "cpu")
        torch.set_default_device("cpu")
        torch.manual_seed(seed)
        random.seed(seed)

        self.applyRandomInjectors(testcaseName)
        cpuReturnValues = self.runner.run(suite).getReturnValues()
        if testcaseName in cpuReturnValues:
            cpuResult = cpuReturnValues[testcaseName]
        else:
            print(f"[ERROR] Error run {testcaseName} on cpu, return -1")
            return -1
        for record in self.storage.values():
            record["status"] = True

        if debug:
            print(f"[DEBUG] result on cpu is \n{cpuResult}")

        deviceResult = None
        passed = False
        if device:
            torch.set_default_device(device)
            suite = self.loader.loadTestsFromTestCase(testcase)
            self.sendValueToDevice(testcaseName, self.storage, device)
            torch.manual_seed(seed)
            random.seed(seed)

            deviceReturnValues = self.runner.run(suite).getReturnValues()
            if testcaseName in deviceReturnValues:
                deviceResult = deviceReturnValues[testcaseName]
            else:
                print(f"[ERROR] Error run {testcaseName} on {device}, return -1")
                return -1
            if device == "cpu":
                print(f"[DEVICE TESTING REMINDER] Don't forget to test on device, or it will be lack of compatibility.")
                # print(f"[DEBUG] result on {device} is \n{deviceResult}")    
            else:
                if debug:
                    print(f"[DEBUG] result on {device} is \n{deviceResult}")

        if cpuResult is None and deviceResult is None:
            print(f"[WARN] Both CPU and Device have no return, if normal ignore this, defaultly Passed.")
            return True
        elif cpuResult is None or deviceResult is None:
            print(f"[ERROR] One of CPU or Device has no return, there must be something wrong.")
        else:
            if cpuResult is not None and deviceResult is not None:
                if torch.is_tensor(deviceResult):
                    cpuResult = cpuResult.to(torch.device("cpu"))
                    deviceResult = deviceResult.to(torch.device("cpu"))

                # Comparison
                passed = self.judgeCommon(cpuResult, deviceResult, testcaseName, device)

                if passed and debug:
                    print(testcaseName + f" has passed the test on {device}, return True\n\n\n")
                elif not passed:
                    print(f"[WARN] {testcaseName} has not passed the test on {device}, return False\n\n\n")

        self.resetRandom()
        self.resetTester()
        return passed

    def applyRandomInjectors(self, testcaseName: str) -> None:
        """
        **description**
        Apply the randomInjector decorator to random functions.

        **params**
        testcaseName (str): The name of the test case.
        """
        setattr(random, 'randint', randomInjector(self.rrandint, self.storage, testcaseName))
        setattr(random, 'uniform', randomInjector(self.uniform, self.storage, testcaseName))
        setattr(torch, 'randn', randomInjector(self.randn, self.storage, testcaseName))
        setattr(torch, 'normal', randomInjector(self.normal, self.storage, testcaseName))
        setattr(torch, 'rand', randomInjector(self.rand, self.storage, testcaseName))
        setattr(torch, "randint", randomInjector(self.trandint, self.storage, testcaseName))
        setattr(torch, "randperm", randomInjector(self.rrandperm, self.storage, testcaseName))
        setattr(random, 'choice', randomInjector(self.rchoice, self.storage, testcaseName))
        setattr(random, 'random', randomInjector(self.rrandom, self.storage, testcaseName))
        setattr(random, 'shuffle', randomInjector(self.rshuffle, self.storage, testcaseName))


    def injectModule(self, module: type, testcaseName: str) -> None:
        """
        **description**
        Inject the randomInjector into a module's attributes.

        **params**
        module (type): The module containing attributes to be injected.
        testcaseName (str): The name of the test case.
        """
        for attr in getAttributes(module):
            obj = getattr(module, attr)
            setattr(torch.nn, obj.__name__, randomInjector(obj, self.storage, testcaseName))

    def judgeCommon(self, cpuResult, deviceResult, testcaseName, device):
        """
        **description**
        Compares the results from the CPU and the specified device to determine if they match.

        **params**
        cpuResult: The result from the CPU run.
        deviceResult: The result from the device run.
        testcaseName (str): The name of the test case.

        **returns**
        passed (bool): True if the results match, False otherwise.

        **raises**
        ValueError: If an error occurs during comparison, providing the test case name.
        """
        torch.set_default_device("cpu")
        cpu = torch.device("cpu")
        passed = False
        try:
            if isinstance(cpuResult, object):
                passed = str(cpuResult) == str(deviceResult)
            if isinstance(cpuResult, bool):
                passed = cpuResult == deviceResult

            if type(cpuResult) == type(1) or type(cpuResult) == type(3.14):
                passed = np.allclose(cpuResult, deviceResult, rtol=1e-05, atol=1e-06)


            if torch.is_tensor(cpuResult):
                cpuResult = cpuResult.to(cpu)
                deviceResult = deviceResult.to(cpu)
                try:
                    passed = torch.allclose(cpuResult.to(cpu), deviceResult.to(cpu), rtol=1e-05, atol=1e-06, equal_nan=True)#  or str(cpuResult.to(cpu)) == str(deviceResult.to(cpu))
                except Exception as e:
                    passed = str(cpuResult.to(cpu)) == str(deviceResult.to(cpu))


            if isinstance(cpuResult, (tuple, list)):
                for idx in range(len(cpuResult)):
                    if isinstance(cpuResult, object):
                        passed = str(cpuResult[idx]) == str(deviceResult[idx])
                    if not type(cpuResult[idx]) == type(deviceResult[idx]):
                        return False
                    if isinstance(cpuResult[idx], bool):
                        passed = cpuResult[idx] == deviceResult[idx]
                        if not passed: return False
                    if torch.is_tensor(cpuResult[idx]):
                        passed = torch.allclose(cpuResult[idx].to(cpu), deviceResult[idx].to(cpu), rtol=1e-05, atol=1e-06, equal_nan=True)
                        if not passed: return False

            if not passed:
                passed = str(cpuResult) == str(deviceResult).replace(f", device='{device}:0'", "")

        except Exception as e:
            raise ValueError(f"The testcase that cause the comparison error is `{testcaseName}`") from e
        return passed

    def sendValueToDevice(self, testcaseName: str, storage: dict, device: str) -> None:
        """
        **description**
        Send the values from storage to the specified device.

        **params**
        testcaseName (str): The name of the test case.
        storage (dict): The storage dictionary containing intermediate values.
        device (str): The device to which values should be sent (e.g., 'cuda').
        """
        if testcaseName in storage:
            device = torch.device(device)
            for lst in storage[testcaseName]["result"].values():
                for idx, val in enumerate(lst):
                    if torch.is_tensor(val):
                        lst[idx] = val.to(device)
        else:
            pass

    def resetTester(self) -> None:
        """
        **description**
        Resets the tester by reinitializing the loader, runner, and storage.

        **params**
        None
        """
        self.loader = MyTestLoader()
        self.runner = MyTestRunner(verbosity=2)
        self.storage = {}

    def resetRandom(self) -> None:
        """
        **description**
        Resets the random functions to their original state.

        **params**
        None
        """
        setattr(random, 'randint', self.rrandint)
        setattr(random, 'uniform', self.uniform)
        setattr(torch, 'randn', self.randn)
        setattr(torch, 'normal', self.normal)
        setattr(torch, 'rand', self.rand)
        setattr(torch, 'randint', self.trandint)
        setattr(random, 'choice', self.rchoice)
        setattr(random, 'random', self.rrandom)
        setattr(random, 'shuffle', self.rshuffle)
        setattr(torch, 'randperm', self.rrandperm)
