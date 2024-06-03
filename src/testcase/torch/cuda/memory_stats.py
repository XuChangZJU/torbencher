
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_stats)
class TorchCudaMemoryStatsTestCase(TorBencherTestCaseBase):
    def test_memory_stats_0(self, input=None):
        if input is not None:
            result = torch.cuda.memory_stats(input[0])
            return result
        a = 0
        result = torch.cuda.memory_stats(a)
        return result
    def test_memory_stats_1(self, input=None):
        if input is not None:
            result = torch.cuda.memory_stats(device=input[0])
            return result
        a = 0
        result = torch.cuda.memory_stats(device=a)
        return result

