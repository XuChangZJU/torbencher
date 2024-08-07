import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.solve_triangular)
class TorchLinalgSolveUtriangularTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_solve_triangular_correctness(self):
        # Randomly determine the dimensions of the matrices
        n = random.randint(2, 5)  # Random size for the square matrix A
        k = random.randint(1, 5)  # Random number of columns for matrix B

        # Generate a random upper triangular matrix A
        A = torch.randn(n, n).triu_()

        # Generate a random matrix B
        B = torch.randn(n, k)

        # Solve the triangular system AX = B
        X = torch.linalg.solve_triangular(A, B, upper=True)

        # Verify the solution
        assert torch.allclose(A @ X, B, atol=1e-6), "The solution X does not satisfy AX = B"

        return X
