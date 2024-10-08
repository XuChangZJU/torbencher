import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.device_count)
class TorchCpuDeviceUcountTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpu_device_count_correctness(self):
        # No input parameters for torch.cpu.device_count
        number_of_cpu_devices = torch.cpu.device_count()
        return number_of_cpu_devices
