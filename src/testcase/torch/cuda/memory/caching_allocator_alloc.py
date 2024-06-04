

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.cuda.memory.caching_allocator_alloc)
class TorchCudaMemoryCachingAllocatorAllocTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_caching_allocator_alloc(self):
        
        a = 100
        b = torch.device('cuda:0')
        c = 0
        result = torch.cuda.memory.caching_allocator_alloc(a, b, c)
        return result
