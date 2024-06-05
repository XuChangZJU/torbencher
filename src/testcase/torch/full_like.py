
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.full_like)
class TorchFullLikeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full_like_correctness(self):
        input = torch.randn(random.randint(1, 10))
        fill_value = random.uniform(0.1, 10.0)
        result = torch.full_like(input, fill_value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_full_like_large_scale(self):
        input = torch.randn(random.randint(1000, 10000))
        fill_value = random.uniform(0.1, 10.0)
        result = torch.full_like(input, fill_value)
        return result

