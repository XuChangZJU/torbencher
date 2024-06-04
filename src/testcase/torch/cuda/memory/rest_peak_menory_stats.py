
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.cuda.memory.reset_peak_memory_stats)
class TorchCudaMemoryResetPeakMemoryStatsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reset_peak_memory_stats_0(self):
        
        a = torch.device('cuda')
        result = torch.cuda.memory.reset_peak_memory_stats(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reset_peak_memory_stats_1(self):
        
        a = torch.device('cuda')
        result = torch.cuda.memory.reset_peak_memory_stats(device=a)
        return result