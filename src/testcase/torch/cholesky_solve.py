import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cholesky_solve)
class TorchCholeskyUsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cholesky_solve_correctness(self):
        # Random dimension for the tensors
        batch_dims = random.randint(0, 2)

        n = random.randint(2, 5)  # Random size for the n x n (and n x k) matrix
        k = random.randint(1, 5)  # Random size for the n x k matrix

        # Generate random positive-definite matrix A with shape (*, n, n)
        A_shape = [n, n] if batch_dims == 0 else [random.randint(1, 3) for _ in range(batch_dims)] + [n, n]
        A = torch.randn(A_shape)
        A = A @ A.transpose(-2, -1) + torch.eye(n, device= A.device) * 1e-3

        # Extract Cholesky decomposition
        L = torch.linalg.cholesky(A)

        # Generate B tensor with shape (*, n, k)
        B_shape = [n, k] if batch_dims == 0 else A_shape[:-1] + [k]
        B = torch.randn(B_shape)

        # Compute cholesky_solve
        result = torch.cholesky_solve(B, L)
        return result
