import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.cholesky)
class TorchLinalgCholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_cholesky_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 10)
        # Generate a random positive-definite matrix A
        A = torch.randn(dim, dim)
        A = A @ A.T + torch.eye(dim)  # Make A positive-definite
        # Calculate the Cholesky decomposition
        L = torch.linalg.cholesky(A)
        # Return the result
        return L
