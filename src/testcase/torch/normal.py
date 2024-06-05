
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.normal)
class TorchNormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normal_correctness(self):
        mean = torch.randn(random.randint(1, 10))
        std = torch.rand(random.randint(1, 10))
        result = torch.normal(mean, std)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_normal_large_scale(self):
        mean = torch.randn(random.randint(1000, 10000))
        std = torch.rand(random.randint(1000, 10000))
        result = torch.normal(mean, std)
        return result

