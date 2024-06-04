
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.max_memory_allocated)
class TorchCudaMaxMemoryAllocatedTestCase(TorBencherTestCaseBase):
    @test_api_version.less_than("1.9.0")
    def test_max_memory_allocated(self):
        a = 0
        result = torch.cuda.max_memory_allocated(a)
        return result

