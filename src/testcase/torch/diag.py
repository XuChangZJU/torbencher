import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.diag)
class TorchDiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_diag_correctness_vector(self):
        # Randomly generate the size of the vector
        vector_length = random.randint(1, 10)

        # Generate a random vector of the given size
        vector = torch.randn(vector_length)

        # Compute the diagonal matrix from the vector
        result = torch.diag(vector)
        return result

    def test_diag_correctness_matrix(self):
        # Randomly generate the size of the matrix
        matrix_size = random.randint(2, 10)

        # Generate a random matrix of the given size
        matrix = torch.randn(matrix_size, matrix_size)

        # Generate a random value for the diagonal index
        diagonal_idx = random.randint(-matrix_size + 1, matrix_size - 1)

        # Compute the diagonal elements tensor at the given index
        result = torch.diag(matrix, diagonal_idx)
        return result

    # Example calls to the test functions
    print()
    print()
