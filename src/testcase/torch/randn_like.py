
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randn_like)
class TorchRandnLikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randn_like_correctness(self):
        input = torch.randn(random.randint(1, 10))
        result = torch.randn_like(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_randn_like_large_scale(self):
        input = torch.randn(random.randint(1000, 10000))
        result = torch.randn_like(input)
        return result

