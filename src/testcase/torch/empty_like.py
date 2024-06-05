
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.empty_like)
class TorchEmptyLikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_empty_like_correctness(self):
        input = torch.randn(random.randint(1, 10))
        result = torch.empty_like(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_empty_like_large_scale(self):
        input = torch.randn(random.randint(1000, 10000))
        result = torch.empty_like(input)
        return result

