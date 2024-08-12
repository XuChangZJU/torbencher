import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.amax)
class TorchTensorAmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_amax_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Calculate the maximum value of the input tensor along all dimensions
        output_tensor = input_tensor.amax()

        # Return the output tensor
        return output_tensor
