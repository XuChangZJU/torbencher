import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.is_available)
class TorchCpuIsUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpu_is_available_correctness(self):
        # No specific parameters to randomize for torch.cpu.is_available()
        result_cpu_availability = torch.cpu.is_available()
        return result_cpu_availability
