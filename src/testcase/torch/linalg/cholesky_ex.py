
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.cholesky_ex)
class TorchLinalgCholeskyExTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_cholesky_ex_correctness(self):
        dim = random.randint(2, 10)
        A = torch.randn(dim, dim)
        A = A @ A.T  # Ensure A is positive-definite
        result = torch.linalg.cholesky_ex(A)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_cholesky_ex_large_scale(self):
        dim = random.randint(100, 1000)
        A = torch.randn(dim, dim)
        A = A @ A.T
        result = torch.linalg.cholesky_ex(A)
        return result

