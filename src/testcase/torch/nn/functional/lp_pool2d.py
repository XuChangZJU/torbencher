import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.lp_pool2d)
class TorchNnFunctionalLpUpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lp_pool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        channels = random.randint(1, 4)  # Random number of channels
        height = random.randint(5, 10)  # Random height of the input tensor
        width = random.randint(5, 10)  # Random width of the input tensor

        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, height, width)

        # Randomly generate the power parameter p
        p = 1

        # Randomly generate the kernel size
        kernel_size = random.randint(2, 4)

        # Randomly generate the stride
        stride = random.randint(1, 3)

        # Apply lp_pool2d with the generated parameters
        result = torch.nn.functional.lp_pool2d(input_tensor, p, kernel_size, stride)
        return result
