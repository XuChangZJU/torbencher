
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky_solve)
class TorchCholeskySolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim, dim)
        A = torch.randn(dim, dim)
        result = torch.cholesky_solve(A, input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_large_scale(self):
        dim = random.randint(100, 1000)
        input = torch.randn(dim, dim)
        A = torch.randn(dim, dim)
        result = torch.cholesky_solve(A, input)
        return result

