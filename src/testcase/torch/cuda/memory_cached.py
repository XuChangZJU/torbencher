
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_cached)
class TorchCudaMemoryCachedTestCase(TorBencherTestCaseBase):
    @test_api_version.less_than("1.9.0")
    def test_memory_cached(self, input=None):
        if input is not None:
            result = torch.cuda.memory_cached(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.memory_cached(a)
        return [result, [a]]

