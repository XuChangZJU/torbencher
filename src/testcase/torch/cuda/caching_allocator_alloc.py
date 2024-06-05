
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.caching_allocator_alloc)
class TorchCudaCachingAllocatorAllocTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_alloc_correctness(self):
        size = random.randint(1, 10)
        result = torch.cuda.caching_allocator_alloc(size)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_alloc_large_scale(self):
        size = random.randint(1000, 10000)
        result = torch.cuda.caching_allocator_alloc(size)
        return result

