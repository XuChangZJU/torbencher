import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.matrix_exp)
class TorchTensorMatrixUexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_matrix_exp_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the square matrix (minimum 2 to ensure matrix properties)
        num_of_elements_each_dim = random.randint(2,
                                                  5)  # Random number of elements each dimension (minimum 2 for square matrix)
        input_size = [num_of_elements_each_dim, num_of_elements_each_dim]  # Ensuring the tensor is a square matrix

        tensor = torch.randn(input_size)  # Generating a random square matrix
        result = tensor.matrix_exp()  # Applying the matrix exponential function
        return result
