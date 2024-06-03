
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.max_memory_reserved)
class TorchCudaMaxMemoryReservedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_max_memory_reserved(self, input=None):
        if input is not None:
            result = torch.cuda.max_memory_reserved(input[0])
            return result
        a = 0
        result = torch.cuda.max_memory_reserved(a)
        return result

