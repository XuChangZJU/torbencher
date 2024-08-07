import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.matrix_power)
class TorchTensorMatrixUpowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matrix_power_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the square matrix (must be at least 2x2)
        matrix_size = [dim, dim]  # Square matrix dimensions

        tensor = torch.randn(matrix_size)  # Random square matrix
        power = random.randint(1, 5)  # Random power between 1 and 5

        result = tensor.matrix_power(power)
        return result
