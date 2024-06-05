
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.eigvals)
class TorchLinalgEigvalsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_eigvals_correctness(self):
        dim = random.randint(2, 10)
        A = torch.randn(dim, dim)
        result = torch.linalg.eigvals(A)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_eigvals_large_scale(self):
        dim = random.randint(100, 1000)
        A = torch.randn(dim, dim)
        result = torch.linalg.eigvals(A)
        return result

