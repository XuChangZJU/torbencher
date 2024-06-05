
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arcsin)
class TorchArcsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arcsin_correctness(self):
        input = torch.rand(random.randint(1, 10)) * 2 - 1
        result = torch.arcsin(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_arcsin_large_scale(self):
        input = torch.rand(random.randint(1000, 10000)) * 2 - 1
        result = torch.arcsin(input)
        return result

