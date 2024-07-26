import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cholesky_inverse)
class TorchCholeskyinverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_inverse_correctness(self):
        # Randomly generate matrix dimension sizes
        batch_dim = random.randint(0, 3)  # Random number of batch dimensions (0 to 3)
        n = random.randint(2, 5)  # Random size of the matrix (must be at least 2x2)

        batch_sizes = [random.randint(1, 3) for _ in range(batch_dim)]  # Random sizes for batch dimensions
        matrix_size = batch_sizes + [n, n]

        # Create a symmetric positive-definite matrix A by using random values
        A = torch.randn(matrix_size)
        A = A @ A.transpose(-1, -2) + torch.eye(n) * 1e-3  # Symmetric positive-definite matrix

        # Compute the Cholesky decomposition of A
        L = torch.linalg.cholesky(A)

        # Compute the inverse using the cholesky_inverse function
        result = torch.cholesky_inverse(L)

        return result
