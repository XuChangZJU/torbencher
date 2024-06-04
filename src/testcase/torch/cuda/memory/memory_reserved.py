
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory.memory_reserved)
class TorchCudaMemoryMemoryReservedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_memory_reserved(self):
        
        a = torch.device('cuda')
        result = torch.cuda.memory.memory_reserved(a)
        return result
