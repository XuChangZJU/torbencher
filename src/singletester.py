import torch;
import random;

from .util.apitools import *;
from .testcase.TorBencherTestCaseBase import TorBencherTestCaseBase;
from .util.unitest import MyTestRunner, MyTestLoader;
from .util.decorator import randomInjector;
# from torch import nn;

class SingleTester:
    """
    **description**
    SingleTester is a class responsible for running test cases derived from TorBencherTestCaseBase on both CPU and optionally a specified device (like GPU) while ensuring consistent random seeds.

    **Attributes**
    loader (MyTestLoader): Instance for loading test cases.
    runner (MyTestRunner): Instance for running test suites.
    storage (dict): Dictionary to store intermediate values for random number injections.

    **Methods**
    run(testcase: TorBencherTestCaseBase, device: str = None, seed: int = 443) -> None:
        Runs the provided test case on CPU and optionally on a specified device, comparing the results.
    applyRandomInjectors(testcaseName: str) -> None:
        Apply the randomInjector decorator to random functions.
    sendValueToDevice(testcaseName: str, storage: dict, device: str) -> None:
        Send the values from storage to the specified device.
    """

    def __init__(self):
        self.loader = MyTestLoader();
        self.runner = MyTestRunner();
        self.storage = {};

    def run(self, testcase: TorBencherTestCaseBase, device: str = None, seed: int = 443) -> None:
        """
        **description**
        Runs the provided test case on CPU and optionally on a specified device, comparing the results.

        **params**
        testcase (TorBencherTestCaseBase): The test case class to be tested.
        device (str, optional): The device to test on (e.g., 'cuda'). Defaults to None.
        seed (int, optional): The seed for random number generation. Defaults to 443.

        **raises**
        AssertionError: If the provided testcase is not a subclass of TorBencherTestCaseBase.
        Warning/Error: If one of CPU or Device has no return.
        """
        assert issubclass(testcase, TorBencherTestCaseBase);
        testcaseName = testcase.__name__;
        print(f"[INITIALIZE] Start testing {testcaseName}");

        suite = self.loader.loadTestsFromTestCase(testcase);
        torch.set_default_device("cpu");
        torch.manual_seed(seed);
        random.seed(seed);

        self.applyRandomInjectors(testcaseName);

        cpuResult = self.runner.run(suite).getReturnValues()[testcaseName][0];
        for record in self.storage.values():
            record["status"] = True;

        print(f"[DEBUG] result on cpu is {cpuResult}");

        deviceResult = None;
        if device:
            torch.set_default_device(device);
            suite = self.loader.loadTestsFromTestCase(testcase);
            self.sendValueToDevice(testcaseName, self.storage, device);
            torch.manual_seed(seed);
            random.seed(seed);

            deviceResult = self.runner.run(suite).getReturnValues()[testcaseName][0];
            print(f"[DEBUG] result on {device} is {deviceResult}");
        else:
            print(f"[DEVICE TESTING REMINDER] Don't forget to test on device, or it will return None here");

        if cpuResult is None and deviceResult is None:
            print(f"[WARN] Both CPU and Device have no return, if normal ignore this.");
        elif cpuResult is None or deviceResult is None:
            print(f"[ERROR] One of CPU or Device has no return, there must be something wrong.");
        else:
            if cpuResult is not None and deviceResult is not None:
                if torch.is_tensor(deviceResult):
                    deviceResult = deviceResult.to(torch.device("cpu"));
                passed = torch.allclose(cpuResult, deviceResult);
                if passed:
                    print(testcaseName + " has passed the test");
                else:
                    print(f"[WARN] {testcaseName} has not passed the test");

    def applyRandomInjectors(self, testcaseName: str) -> None:
        """
        **description**
        Apply the randomInjector decorator to random functions.

        **params**
        testcaseName (str): The name of the test case.
        """
        setattr(random, 'randint', randomInjector(random.randint, self.storage, testcaseName));
        setattr(random, 'uniform', randomInjector(random.uniform, self.storage, testcaseName));
        setattr(torch, 'randn', randomInjector(torch.randn, self.storage, testcaseName));
        # self.injectModule(nn, testcaseName);

    def injectModule(self, module, testcaseName):
        for attr in getAttributes(module):
            obj = getattr(module, attr);
            setattr(torch.nn, obj.__name__, randomInjector(obj, self.storage, testcaseName));

    def sendValueToDevice(self, testcaseName: str, storage: dict, device: str) -> None:
        """
        **description**
        Send the values from storage to the specified device.

        **params**
        testcaseName (str): The name of the test case.
        storage (dict): The storage dictionary containing intermediate values.
        device (str): The device to which values should be sent (e.g., 'cuda').
        """
        device = torch.device(device);
        for lst in storage[testcaseName]["result"].values():
            for idx, val in enumerate(lst):
                if torch.is_tensor(val):
                    lst[idx] = val.to(device);
