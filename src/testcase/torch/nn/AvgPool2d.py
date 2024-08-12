import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.AvgPool2d)
class TorchNnAvgpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_avgpool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 10)  # Batch size
        C = random.randint(1, 10)  # Number of channels
        H_in = random.randint(10, 20)  # Height of the input tensor
        W_in = random.randint(10, 20)  # Width of the input tensor

        # Randomly generate kernel size, stride, and padding
        kernel_size = (random.randint(4, 5), random.randint(4, 5))
        stride = (random.randint(1, 3), random.randint(1, 3))
        padding = (random.randint(0, 2), random.randint(0, 2))

        # Create the AvgPool2d layer with the generated parameters
        avg_pool = torch.nn.AvgPool2d(kernel_size, stride, padding)

        # Generate a random input tensor with the specified dimensions
        input_tensor = torch.randn(N, C, H_in, W_in)

        # Apply the AvgPool2d layer to the input tensor
        output_tensor = avg_pool(input_tensor)

        return output_tensor
