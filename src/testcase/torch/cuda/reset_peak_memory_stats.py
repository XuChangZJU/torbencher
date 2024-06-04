
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.reset_peak_memory_stats)
class TorchCudaResetPeakMemoryStatsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_reset_peak_memory_stats(self):
        a = torch.device('cuda')
        result = torch.cuda.reset_peak_memory_stats(a)
        return result

