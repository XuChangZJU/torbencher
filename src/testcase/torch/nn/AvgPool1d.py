import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.AvgPool1d)
class TorchNnAvgpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_avgpool1d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        num_channels = random.randint(1, 4)  # Random number of channels
        length = random.randint(5, 10)  # Random length of the input signal

        # Randomly generate parameters for AvgPool1d
        kernel_size = random.randint(4, 5)  # Random kernel size
        stride = random.randint(1, kernel_size)  # Random stride, ensuring it's valid
        padding = random.randint(0, 2)  # Random padding

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, num_channels, length)

        # Create the AvgPool1d layer with the generated parameters
        avg_pool = torch.nn.AvgPool1d(kernel_size, stride, padding)

        # Apply the AvgPool1d layer to the input tensor
        result = avg_pool(input_tensor)
        return result
