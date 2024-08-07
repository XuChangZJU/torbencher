import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.current_device)
class TorchCpuCurrentdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cpu_current_device_correctness(self):
        # No specific input parameters needed for torch.cpu.current_device
        current_device = torch.cpu.current_device()
        return current_device
