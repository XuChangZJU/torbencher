import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.subtract)
class TorchSubtractTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_subtract_correctness(self):
        # Randomly generate the dimension of the tensors
        dim = random.randint(1, 4)
        # Randomly generate the number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input_size as a list
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate two random tensors with the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
        # Calculate the result of subtract
        result = torch.subtract(input_tensor, other_tensor)
        return result
