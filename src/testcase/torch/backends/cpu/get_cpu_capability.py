import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cpu.get_cpu_capability)
class TorchBackendsCpuGetUcpuUcapabilityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_get_cpu_capability_correctness(self):
        # No input parameters needed for torch.backends.cpu.get_cpu_capability
        result = torch.backends.cpu.get_cpu_capability()
        return result
