
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.device_of)
class TorchCudaDeviceOfTestCase(TorBencherTestCaseBase):
    def test_device_of_correctness(self):
        tensor = torch.randn(random.randint(1, 10))
        result = torch.cuda.device_of(tensor)
        return result

    def test_device_of_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000))
        result = torch.cuda.device_of(tensor)
        return result

