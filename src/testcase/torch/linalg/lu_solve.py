import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.lu_solve)
class TorchLinalgLuUsolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lu_solve_correctness(self):
        # Random dimension for the square matrix A
        n = random.randint(2, 5)
        # Random number of columns for matrix B
        k = random.randint(1, 5)

        # Generate a random square matrix A and perform LU factorization
        A = torch.randn(n, n)
        LU, pivots = torch.lu(A)

        # Generate a random matrix B with compatible dimensions
        B = torch.randn(n, k)

        # Solve the linear system using the LU decomposition
        X = torch.lu_solve(B, LU, pivots)

        # Verify the solution by checking if A @ X is close to B
        assert torch.allclose(A @ X, B, atol=1e-6), "The solution X does not satisfy AX = B"

        return X
