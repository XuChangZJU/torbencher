import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.det)
class TorchTensorDetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_det_correctness(self):
        # Random dimension for the square matrix (2x2 or 3x3 for simplicity)
        dim = random.randint(2, 3)

        # Generate a random square matrix of the chosen dimension
        matrix_size = [dim, dim]
        random_matrix = torch.randn(matrix_size)

        # Compute the determinant of the matrix
        result = random_matrix.det()
        return result
