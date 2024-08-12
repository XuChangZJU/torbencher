import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.matrix_power)
class TorchLinalgMatrixUpowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_matrix_power_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random square matrix
        matrix = torch.randn(input_size + input_size)
        # Generate a random integer exponent
        n = random.randint(-5, 5)
        # Calculate the matrix power
        result = torch.linalg.matrix_power(matrix, n)
        # Return the result
        return result
