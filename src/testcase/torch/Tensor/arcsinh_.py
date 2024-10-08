import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.arcsinh_)
class TorchTensorArcsinhUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_arcsinh__correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Perform the in-place arcsinh operation
        input_tensor.arcsinh_()
        # Return the tensor after the in-place operation
        return input_tensor
