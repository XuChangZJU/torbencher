import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cpu.get_cpu_capability)
class TorchBackendsCpuGetcpucapabilityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_cpu_capability_correctness(self):
        # No input parameters needed for torch.backends.cpu.get_cpu_capability
        result = torch.backends.cpu.get_cpu_capability()
        return result
    