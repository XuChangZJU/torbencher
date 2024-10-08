import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.lu)
class TorchLuTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lu_correctness(self):
        # Randomly generate dimensions for the matrix
        m = random.randint(2, 5)
        n = random.randint(2, 5)

        # Generate a random matrix (tensor) of size (m, n)
        A = torch.randn(m, n)

        # Perform LU factorization
        LU, pivots = torch.lu(A)

        return LU, pivots

    def test_lu_with_infos(self):
        # Randomly generate dimensions for the matrix
        m = random.randint(2, 5)
        n = random.randint(2, 5)

        # Generate a random matrix (tensor) of size (m, n)
        A = torch.randn(m, n)

        # Perform LU factorization and get info
        LU, pivots, info = torch.lu(A, get_infos=True)

        return LU, pivots, info
