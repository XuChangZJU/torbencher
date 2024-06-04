
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.caching_allocator_delete)
class TorchCudaCachingAllocatorDeleteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_caching_allocator_delete(self):
        a = torch.cuda.caching_allocator_alloc(100)
        result = torch.cuda.caching_allocator_delete(a)
        return result

