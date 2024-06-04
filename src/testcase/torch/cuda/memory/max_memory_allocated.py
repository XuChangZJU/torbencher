
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.memory.max_memory_allocated)
class TorchCudaMemoryMaxMemoryAllocatedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_memory_allocated_0(self):
        
        a = 0
        result = torch.cuda.memory.max_memory_allocated(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_memory_allocated_1(self):
        
        a = 0
        result = torch.cuda.memory.max_memory_allocated(device=a)
        return result
