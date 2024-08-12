import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.current_device)
class TorchCpuCurrentUdeviceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpu_current_device_correctness(self):
        # No specific input parameters needed for torch.cpu.current_device
        current_device = torch.cpu.current_device()
        return current_device
