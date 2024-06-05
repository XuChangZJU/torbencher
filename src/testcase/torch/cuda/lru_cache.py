
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.lru_cache)
class TorchCudaLruCacheTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_lru_cache_correctness(self):
        result = torch.cuda.lru_cache()
        return result

    @test_api_version.larger_than("1.10.0")
    def test_lru_cache_large_scale(self):
        result = torch.cuda.lru_cache()
        return result

