

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.memory.caching_allocator_delete)
class TorchCudaMemoryCachingAllocatorDeleteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_caching_allocator_delete(self, input=None):
        if input is not None:
            result = torch.cuda.memory.caching_allocator_delete(input[0])
            return result
        a = torch.cuda.memory.caching_allocator_alloc(100)
        result = torch.cuda.memory.caching_allocator_delete(a)
        return result
