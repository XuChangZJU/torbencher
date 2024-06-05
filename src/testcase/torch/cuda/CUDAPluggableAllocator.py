
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.CUDAPluggableAllocator)
class TorchCudaCUDAPluggableAllocatorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_cudapluggableallocator_correctness(self):
        allocator = torch.cuda.CUDAPluggableAllocator()
        result = allocator.allocator
        return result

    @test_api_version.larger_than("1.10.0")
    def test_cudapluggableallocator_large_scale(self):
        allocator = torch.cuda.CUDAPluggableAllocator()
        result = allocator.allocator
        return result

