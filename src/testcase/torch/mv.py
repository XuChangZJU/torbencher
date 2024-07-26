import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.mv)
class TorchMvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mv_correctness(self):
        # Define the dimensions of the matrix and vector
        n = random.randint(1, 10)  # Number of rows in the matrix
        m = random.randint(1, 10)  # Number of columns in the matrix and size of the vector

        # Generate random matrix and vector with the specified dimensions
        matrix = torch.randn(n, m)
        vector = torch.randn(m)

        # Perform matrix-vector multiplication
        result = torch.mv(matrix, vector)
        return result
