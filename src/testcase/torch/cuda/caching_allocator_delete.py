
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.caching_allocator_delete)
class TorchCudaCachingAllocatorDeleteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_delete_correctness(self):
        size = random.randint(1, 10)
        ptr = torch.cuda.caching_allocator_alloc(size)
        result = torch.cuda.caching_allocator_delete(ptr)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_delete_large_scale(self):
        size = random.randint(1000, 10000)
        ptr = torch.cuda.caching_allocator_alloc(size)
        result = torch.cuda.caching_allocator_delete(ptr)
        return result

