
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.celu)
class TorchCeluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        alpha = random.uniform(0.1, 10.0)
        result = torch.celu(input, alpha=alpha)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_celu_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randn(dim)
        alpha = random.uniform(0.1, 10.0)
        result = torch.celu(input, alpha=alpha)
        return result

