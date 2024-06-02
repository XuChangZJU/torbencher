
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.max_memory_cached)
class TorchCudaMaxMemoryCachedTestCase(TorBencherTestCaseBase):

    def test_max_memory_cached_0(self, input=None):
        if input is not None:
            result = torch.cuda.max_memory_cached(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.max_memory_cached(a)
        return [result, [a]]

    def test_max_memory_cached_1(self, input=None):
        if input is not None:
            result = torch.cuda.max_memory_cached(device=input[0])
            return [result, input]
        a = 0
        result = torch.cuda.max_memory_cached(device=a)
        return [result, [a]]
