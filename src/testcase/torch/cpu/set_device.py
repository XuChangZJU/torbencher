import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.set_device)
class TorchCpuSetUdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpu_set_device_correctness(self):
        # No random parameters needed for torch.cpu.set_device as it only accepts device ordinals
        device_index = random.randint(0,
                                      torch.cuda.device_count() - 1) if torch.cuda.is_available() else 0  # Random valid device index
        result = torch.cpu.set_device(device_index)
        return result
