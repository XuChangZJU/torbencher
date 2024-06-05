
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.acos)
class TorchAcosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_acos_correctness(self):
        input = torch.rand(random.randint(1, 10))
        result = torch.acos(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_acos_large_scale(self):
        input = torch.rand(random.randint(1000, 10000))
        result = torch.acos(input)
        return result

