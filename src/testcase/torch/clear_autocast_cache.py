
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clear_autocast_cache)
class TorchClearAutocastCacheTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clear_autocast_cache_correctness(self):
        result = torch.clear_autocast_cache()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clear_autocast_cache_large_scale(self):
        result = torch.clear_autocast_cache()
        return result

