import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.min)
class TorchTensorMinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_Tensor_min_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Call the min function on the input tensor
        result = input_tensor.min()
        # Return the result of the min operation
        return result

    def test_Tensor_min_dim_keepdim_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Generate a random dimension to reduce along
        dim = random.randint(0, len(input_size) - 1)
        # Call the min function on the input tensor with the specified dimension
        result = input_tensor.min(dim)
        # Return the result of the min operation
        return result
