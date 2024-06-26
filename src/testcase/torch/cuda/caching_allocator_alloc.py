
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.caching_allocator_alloc)
class TorchCudaCachingAllocatorAllocTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_alloc_0(self, input=None):
        if input is not None:
            result = torch.cuda.caching_allocator_alloc(input[0])
            return [result, input]
        a = 100
        result = torch.cuda.caching_allocator_alloc(a)
        return [result, [a]]
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_alloc_1(self, input=None):
        if input is not None:
            result = torch.cuda.caching_allocator_alloc(input[0], input[1])
            return [result, input]
        a = 100
        b = 0
        result = torch.cuda.caching_allocator_alloc(a, b)
        return [result, [a, b]]
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_alloc_2(self, input=None):
        if input is not None:
            result = torch.cuda.caching_allocator_alloc(input[0], input[1], input[2])
            return [result, input]
        a = 100
        b = 0
        c = torch.cuda.current_stream()
        result = torch.cuda.caching_allocator_alloc(a, b, c)
        return [result, [a, b, c]]

