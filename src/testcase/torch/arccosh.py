
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arccosh)
class TorchArccoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arccosh_correctness(self):
        input = torch.rand(random.randint(1, 10)) + 1
        result = torch.arccosh(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_arccosh_large_scale(self):
        input = torch.rand(random.randint(1000, 10000)) + 1
        result = torch.arccosh(input)
        return result

