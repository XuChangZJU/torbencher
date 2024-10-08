import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sum)
class TorchTensorSumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sum_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor 
        input_tensor = torch.randn(input_size)
        # Calculate the sum of all elements in the input tensor
        result = input_tensor.sum()
        return result

    def test_sum_with_dim(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Randomly select a dimension to sum over
        dim_to_sum = random.randint(0, len(input_size) - 1)
        # Calculate the sum along the specified dimension
        result = input_tensor.sum(dim=dim_to_sum)
        return result
