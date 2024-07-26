import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cholesky)
class TorchCholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_correctness(self):
        batch_size = random.randint(1, 5)  # Random batch size between 1 and 5
        matrix_dim = random.randint(2, 5)  # Random matrix dimension (minimum 2 to ensure square matrix)

        input_size = (batch_size, matrix_dim, matrix_dim)  # Define the size of the random tensor

        # Create a random tensor and ensure it's symmetric positive-definite by matrix multiplication
        A = torch.randn(input_size)
        A = A @ A.mT + 1e-3 * torch.eye(matrix_dim).expand_as(A)  # Make symmetric positive-definite

        upper = random.choice([True, False])  # Randomly choose upper or lower triangular matrix

        result = torch.cholesky(A, upper)  # Get the Cholesky decomposition

        return result
