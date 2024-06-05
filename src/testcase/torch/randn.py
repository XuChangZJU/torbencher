
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randn)
class TorchRandnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randn_correctness(self):
        size = (random.randint(1, 10),)
        result = torch.randn(size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_randn_large_scale(self):
        size = (random.randint(1000, 10000),)
        result = torch.randn(size)
        return result

