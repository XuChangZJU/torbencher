
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cpu.set_device)
class TorchCpuSetDeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_device_correctness(self):
        device_id = random.randint(0, torch.cuda.device_count() - 1)
        torch.cuda.set_device(device_id)
        return None

    @test_api_version.larger_than("1.1.3")
    def test_set_device_large_scale(self):
        device_id = random.randint(0, torch.cuda.device_count() - 1)
        torch.cuda.set_device(device_id)
        return None


