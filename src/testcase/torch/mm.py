import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.mm)
class TorchMmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_mm_correctness(self):
        # Randomly generate matrix dimensions, ensuring matrix multiplication is valid
        n = random.randint(1, 10)
        m = random.randint(1, 10)
        p = random.randint(1, 10)

        # Generate random input matrices
        input_matrix = torch.randn(n, m)
        mat2_matrix = torch.randn(m, p)

        # Perform matrix multiplication
        result = torch.mm(input_matrix, mat2_matrix)
        return result
