import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.geqrf)
class TorchTensorGeqrfTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_geqrf_correctness(self):
        # Randomly generate the number of rows and columns for the matrix
        num_rows = random.randint(2, 5)
        num_cols = random.randint(2, 5)

        # Generate a random matrix of size (num_rows, num_cols)
        matrix = torch.randn(num_rows, num_cols)

        # Perform QR factorization using geqrf
        q, r = matrix.geqrf()

        return q, r
