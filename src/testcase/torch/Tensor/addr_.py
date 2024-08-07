import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.addr_)
class TorchTensorAddrUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_inplace_correctness(self):
        # Randomly generate dimensions for the matrix
        matrix_rows = random.randint(1, 5)
        matrix_cols = random.randint(1, 5)

        # Generate random tensors for the matrix and vectors
        matrix = torch.randn(matrix_rows, matrix_cols)
        vec1 = torch.randn(matrix_rows)
        vec2 = torch.randn(matrix_cols)

        # Perform the in-place addr operation
        result = matrix.addr_(vec1, vec2)
        return result
