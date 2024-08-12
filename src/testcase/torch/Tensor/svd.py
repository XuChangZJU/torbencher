import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.svd)
class TorchTensorSvdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_svd_correctness(self):
        # Randomly generate dimensions for a 2D tensor (matrix)
        rows = random.randint(2, 5)  # Random number of rows between 2 and 5
        cols = random.randint(2, 5)  # Random number of columns between 2 and 5

        # Generate a random 2D tensor with the specified dimensions
        matrix = torch.randn(rows, cols)

        # Perform Singular Value Decomposition
        U, S, V = matrix.svd()

        return U, S, V
