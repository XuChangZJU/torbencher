
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.current_device)
class TorchCudaCurrentDeviceTestCase(TorBencherTestCaseBase):
    def test_current_device_correctness(self):
        result = torch.cuda.current_device()
        return result

    def test_current_device_large_scale(self):
        result = torch.cuda.current_device()
        return result

