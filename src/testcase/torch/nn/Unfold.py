import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Unfold)
class TorchNnUnfoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unfold_correctness(self):
        # Randomly generate input tensor dimensions
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 5)
        height = random.randint(1, 10)
        width = random.randint(1, 10)

        # Randomly generate kernel size, dilation, padding, and stride
        kernel_size = (
        random.randint(1, height), random.randint(1, width))  # Ensure kernel size is smaller than input dimensions
        dilation = (random.randint(1, 2), random.randint(1, 2))  # Dilation should be a positive integer
        padding = (
        random.randint(0, kernel_size[0] - 1), random.randint(0, kernel_size[1] - 1))  # Padding should be non-negative
        stride = (random.randint(1, height - kernel_size[0] + 1),
                  random.randint(1, width - kernel_size[1] + 1))  # Ensure stride does not exceed input dimensions

        # Create input tensor and Unfold module
        input_tensor = torch.randn(batch_size, channels, height, width)
        unfold = torch.nn.Unfold(kernel_size)

        # Apply Unfold operation
        result = unfold(input_tensor)
        return result
