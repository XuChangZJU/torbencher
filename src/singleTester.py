import random
import numpy as np

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
    run(testcase: TorBencherTestCaseBase, device: str = None, seed: int = 443) -> bool:
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
        self.storage = {}
        self.uniform = random.uniform
        self.rrandint = random.randint
        self.trandint = torch.randint
        self.randn = torch.randn
        self.normal = torch.normal
        self.rand = torch.rand
        self.rchoice = random.choice
        self.rrandom = random.random

    def run(self, testcase: TorBencherTestCaseBase, device: str = "cpu", seed: int = None) -> bool:
        """
        **description**
        Runs the provided test case on CPU and optionally on a specified device, comparing the results.

        **params**
        testcase (TorBencherTestCaseBase): The test case class to be tested.
        device (str, optional): The device to test on (e.g., 'cuda'). Defaults to cpu.
        seed (int, optional): The seed for random number generation. Defaults to 443.

        **returns**
        passed (bool): The passed status of the testcase.

        **raises**
        AssertionError: If the provided testcase is not a subclass of TorBencherTestCaseBase.
        Warning/Error: If one of CPU or Device has no return.
        """
        assert issubclass(testcase, TorBencherTestCaseBase)
        testcaseName = testcase.__name__
        if not seed:
            seed = time.time_ns()
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
        cpuResult = cpuReturnValues[testcaseName] if testcaseName in cpuReturnValues else None
        for record in self.storage.values():
            record["status"] = True

        print(f"[DEBUG] result on cpu is \n{cpuResult}")

        deviceResult = None
        passed = False
        if device == "cpu":
            torch.set_default_device(device)
            suite = self.loader.loadTestsFromTestCase(testcase)
            self.sendValueToDevice(testcaseName, self.storage, device)
            torch.manual_seed(seed)
            random.seed(seed)

            deviceReturnValues = self.runner.run(suite).getReturnValues()
            deviceResult = deviceReturnValues[testcaseName] if testcaseName in deviceReturnValues else None
            if device == "cpu":
		print(f"[DEVICE TESTING REMINDER] Don't forget to test on device, or it will return None here")
                pass
            else:
                print(f"[DEBUG] result on {device} is \n{deviceResult}")
            

        if cpuResult is None and deviceResult is None:
            print(f"[WARN] Both CPU and Device have no return, if normal ignore this, defaultly Passed.")
            return True
        elif cpuResult is None or deviceResult is None:
            print(f"[ERROR] One of CPU or Device has no return, there must be something wrong.")
        else:
            if cpuResult is not None and deviceResult is not None:
                if torch.is_tensor(deviceResult):
                    deviceResult = deviceResult.to(torch.device("cpu"))

                # Comparison
                passed = self.judgeCommon(cpuResult, deviceResult, testcaseName)

                if passed:
                    print(testcaseName + f" has passed the test on {device}\n\n\n")
                else:
                    print(f"[WARN] {testcaseName} has not passed the test on {device}\n\n\n")

        self.resetRandom()
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
        setattr(random, 'choice', randomInjector(self.rchoice, self.storage, testcaseName))
        setattr(random, 'random', randomInjector(self.rrandom, self.storage, testcaseName))

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

    def judgeCommon(self, cpuResult, deviceResult, testcaseName):
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
                passed = type(cpuResult) == type(deviceResult)
            if isinstance(cpuResult, bool):
                passed = cpuResult == deviceResult

            if type(cpuResult) == type(1) or type(cpuResult) == type(3.14):
                passed = np.allclose(cpuResult, deviceResult)

            if torch.is_tensor(cpuResult):
                cpuResult = cpuResult.to(cpu)
                deviceResult = deviceResult.to(cpu)
                passed = torch.allclose(cpuResult, deviceResult)

            if isinstance(cpuResult, tuple):
                for idx in range(len(cpuResult)):
                    if isinstance(cpuResult[idx], bool):
                        passed = cpuResult[idx] == deviceResult[idx]
                        if not passed: return False
                    if torch.is_tensor(cpuResult[idx]):
                        passed = torch.allclose(cpuResult[idx].to(cpu), deviceResult[idx].to(cpu))
                        if not passed: return False
        except Exception as e:
            passed = False
            raise ValueError(f"The testcase that cause the error is `{testcaseName}`") from e
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
