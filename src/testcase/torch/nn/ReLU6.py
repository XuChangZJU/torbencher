import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ReLU6)
class TorchNnRelu6TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ReLU6_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Define ReLU6 module
        relu6 = torch.nn.ReLU6()
        # Calculate output
        output_tensor = relu6(input_tensor)

        return output_tensor
