import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CircularPad2d)
class TorchNnCircularpad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_circular_pad2d_correctness(self):
    # Randomly generate dimensions for the input tensor
    N = random.randint(1, 4)  # Batch size
    C = random.randint(1, 4)  # Number of channels
    H_in = random.randint(3, 10)  # Height of the input tensor
    W_in = random.randint(3, 10)  # Width of the input tensor

    # Randomly generate padding values
    padding_left = random.randint(0, 3)
    padding_right = random.randint(0, 3)
    padding_top = random.randint(0, 3)
    padding_bottom = random.randint(0, 3)
    padding = (padding_left, padding_right, padding_top, padding_bottom)

    # Create a random input tensor
    input_tensor = torch.randn(N, C, H_in, W_in)

    # Apply CircularPad2d
    circular_pad = torch.nn.CircularPad2d(padding)
    result = circular_pad(input_tensor)
    return result
