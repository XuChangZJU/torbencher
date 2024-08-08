import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cpu.current_stream)
class TorchCpuCurrentUstreamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpu_current_stream_correctness(self):
        # No input parameters to randomize

        # Calling torch.cpu.current_stream() to get the current stream
        result = torch.cpu.current_stream()
        return result
