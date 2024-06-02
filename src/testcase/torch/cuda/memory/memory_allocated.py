
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.memory.memory_allocated)
class TorchCudaMemoryMemoryAllocatedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_memory_allocated_0(self, input=None):
        if input is not None:
            result = torch.cuda.memory.memory_allocated(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.memory.memory_allocated(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_memory_allocated_1(self, input=None):
        if input is not None:
            result = torch.cuda.memory.memory_allocated(device=input[0])
            return [result, input]
        a = 0
        result = torch.cuda.memory.memory_allocated(device=a)
        return [result, [a]]

