import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.max_unpool2d)
class TorchNnFunctionalMaxUunpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_max_unpool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        height = random.randint(4, 8)
        width = random.randint(4, 8)

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, channels, height, width)

        # Generate random indices tensor with the same shape as input_tensor
        indices = torch.randint(0, height * width, (batch_size, channels, height, width))

        # Define kernel size, stride, and padding for max pooling
        kernel_size = random.randint(2, 4)
        stride = kernel_size  # To ensure valid unpooling, stride should be equal to kernel_size
        padding = 0  # No padding for simplicity

        # Perform max unpooling
        output = torch.nn.functional.max_unpool2d(input_tensor, indices, kernel_size, stride, padding)

        return output
