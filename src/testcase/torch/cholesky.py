
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky)
class TorchCholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim, dim)
        result = torch.cholesky(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cholesky_large_scale(self):
        dim = random.randint(100, 1000)
        input = torch.randn(dim, dim)
        result = torch.cholesky(input)
        return result

