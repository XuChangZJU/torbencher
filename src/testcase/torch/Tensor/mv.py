import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.mv)
class TorchTensorMvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mv_correctness(self):
        # Generate random dimensions for the matrix and vector
        dim1 = random.randint(1, 4)
        dim2 = random.randint(1, 4)

        # Create random matrix and vector with compatible dimensions for matrix multiplication
        matrix = torch.randn(dim1, dim2)
        vec = torch.randn(dim2)

        # Perform matrix-vector multiplication
        result = matrix.mv(vec)
        return result
