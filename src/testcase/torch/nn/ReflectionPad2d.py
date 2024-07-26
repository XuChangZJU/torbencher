import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ReflectionPad2d)
class TorchNnReflectionpad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reflection_pad2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        H_in = random.randint(5, 10)  # Height of the input tensor
        W_in = random.randint(5, 10)  # Width of the input tensor

        # Randomly generate padding values
        padding_left = random.randint(1, 3)
        padding_right = random.randint(1, 3)
        padding_top = random.randint(1, 3)
        padding_bottom = random.randint(1, 3)
        padding = (padding_left, padding_right, padding_top, padding_bottom)

        # Create a random input tensor
        input_tensor = torch.randn(N, C, H_in, W_in)

        # Apply ReflectionPad2d
        reflection_pad = torch.nn.ReflectionPad2d(padding)
        result = reflection_pad(input_tensor)
        return result
