import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.reciprocal)
class TorchTensorReciprocalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_reciprocal_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of dimensions for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified dimensions
        input_tensor = torch.randn(input_size)
        # Calculate the reciprocal of the input tensor
        result = input_tensor.reciprocal()
        # Return the result tensor
        return result
