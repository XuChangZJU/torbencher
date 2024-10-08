import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.cholesky_ex)
class TorchLinalgCholeskyUexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cholesky_ex_correctness(self):
        # Generate a random dimension for the matrix
        dim = random.randint(1, 10)
        # Generate a random positive-definite matrix
        a = torch.randn(dim, dim)
        a = a @ a.t() + torch.eye(dim) * 1e-6  # Ensure positive-definiteness
        result = torch.linalg.cholesky_ex(a)
        return result
