import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.unfold)
class TorchNnFunctionalUnfoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_unfold_correctness(self):
        # Randomly generate batch size, channels, height, and width for the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        height = random.randint(5, 10)  # Height should be at least as large as kernel size
        width = random.randint(5, 10)  # Width should be at least as large as kernel size

        # Randomly generate kernel size, stride, and padding
        kernel_size = random.randint(1, min(height, width))
        stride = random.randint(1, kernel_size)
        padding = random.randint(0, kernel_size // 2)

        # Create a random 4-D input tensor
        input_tensor = torch.randn(batch_size, channels, height, width)

        # Apply the unfold operation
        result = torch.nn.functional.unfold(input_tensor, kernel_size, stride=stride, padding=padding)
        return result
