import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.any)
class TorchAnyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_any_correctness(self):
        # Define the dimensions of the input tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input sizes for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Create a random tensor of booleans
        input_tensor = torch.randn(input_size) < 0.5
        # Calculate the any reduction of the tensor
        result = torch.any(input_tensor)
        # Return the result
        return result
