
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_default_device)
class TorchSetDefaultDeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_device_correctness(self):
        device = torch.device("cpu")
        result = torch.set_default_device(device)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_default_device_large_scale(self):
        device = torch.device("cuda")
        result = torch.set_default_device(device)
        return result

