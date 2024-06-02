

import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.cuda.memory.caching_allocator_alloc)
class TorchCudaMemoryCachingAllocatorAllocTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_caching_allocator_alloc(self, input=None):
        if input is not None:
            result = torch.cuda.memory.caching_allocator_alloc(input[0], input[1], input[2])
            return [result, input]
        a = 100
        b = torch.device('cuda:0')
        c = 0
        result = torch.cuda.memory.caching_allocator_alloc(a, b, c)
        return [result, [a, b, c]]
