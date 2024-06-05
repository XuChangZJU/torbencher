
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_allocator_backend)
class TorchCudaGetAllocatorBackendTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_get_allocator_backend_correctness(self):
        result = torch.cuda.get_allocator_backend()
        return result

    @test_api_version.larger_than("1.10.0")
    def test_get_allocator_backend_large_scale(self):
        result = torch.cuda.get_allocator_backend()
        return result

