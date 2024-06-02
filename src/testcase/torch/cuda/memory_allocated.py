
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_allocated)
class TorchCudaMemoryAllocatedTestCase(TorBencherTestCaseBase):
    @test_api_version.less_than("1.9.0")
    def test_memory_allocated(self, input=None):
        if input is not None:
            result = torch.cuda.memory_allocated(input[0])
            return [result, input]
        a = 0
        result = torch.cuda.memory_allocated(a)
        return [result, [a]]

