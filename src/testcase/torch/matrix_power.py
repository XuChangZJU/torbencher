import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.matrix_power)
class TorchMatrixUpowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_matrix_power_correctness(self):
        # Generate random dimension for the square matrix
        dim = random.randint(1, 5)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 3)
        # Generate random input size for the square matrix
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input = torch.randn(input_size + input_size)
        # Generate random integer n
        n = random.randint(-5, 5)
        # Calculate matrix power
        result = torch.matrix_power(input, n)
        return result
