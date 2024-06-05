
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.change_current_allocator)
class TorchCudaChangeCurrentAllocatorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_change_current_allocator_correctness(self):
        allocator = torch.cuda.CUDAPluggableAllocator()
        result = torch.cuda.change_current_allocator(allocator)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_change_current_allocator_large_scale(self):
        allocator = torch.cuda.CUDAPluggableAllocator()
        result = torch.cuda.change_current_allocator(allocator)
        return result

