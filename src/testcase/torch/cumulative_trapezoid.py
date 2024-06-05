
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cumulative_trapezoid)
class TorchCumulativeTrapezoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumulative_trapezoid_correctness(self):
        dim = random.randint(1, 10)
        y = torch.randn(dim)
        dx = random.uniform(0.1, 10.0)
        result = torch.cumulative_trapezoid(y, dx=dx)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cumulative_trapezoid_large_scale(self):
        dim = random.randint(1000, 10000)
        y = torch.randn(dim)
        dx = random.uniform(0.1, 10.0)
        result = torch.cumulative_trapezoid(y, dx=dx)
        return result

