
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_device)
class TorchCudaSetDeviceTestCase(TorBencherTestCaseBase):
    def test_set_device_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.set_device(device)
        return result

    def test_set_device_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.set_device(device)
        return result

