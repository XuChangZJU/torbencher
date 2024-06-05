
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.device)
class TorchCudaDeviceTestCase(TorBencherTestCaseBase):
    def test_device_correctness(self):
        device = torch.cuda.device(random.randint(0, torch.cuda.device_count() - 1))
        result = device.type
        return result

    def test_device_large_scale(self):
        device = torch.cuda.device(random.randint(0, torch.cuda.device_count() - 1))
        result = device.type
        return result

