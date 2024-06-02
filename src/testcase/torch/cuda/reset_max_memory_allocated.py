
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.reset_max_memory_allocated)
class TorchCudaResetMaxMemoryAllocatedTestCase(TorBencherTestCaseBase):
    def test_reset_max_memory_allocated_0(self, input=None):
        if input is not None:
            result = torch.cuda.reset_max_memory_allocated(input[0])
            return [result, input]
        a = torch.device('cuda')
        result = torch.cuda.reset_max_memory_allocated(a)
        return [result, [a]]
    def test_reset_max_memory_allocated_1(self, input=None):
        if input is not None:
            result = torch.cuda.reset_max_memory_allocated(device=input[0])
            return [result, input]
        a = torch.device('cuda')
        result = torch.cuda.reset_max_memory_allocated(device=a)
        return [result, [a]]
