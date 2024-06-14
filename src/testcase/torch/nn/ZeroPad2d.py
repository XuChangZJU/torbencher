import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ZeroPad2d)
class TorchNnZeropad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ZeroPad2d_correctness(self):
    # Random input size
    batch_size = random.randint(1, 3)
    channels = random.randint(1, 3)
    height = random.randint(1, 10)
    width = random.randint(1, 10)
    input_size = [batch_size, channels, height, width]

    # Random padding
    padding_left = random.randint(0, 5)
    padding_right = random.randint(0, 5)
    padding_top = random.randint(0, 5)
    padding_bottom = random.randint(0, 5)
    padding = (padding_left, padding_right, padding_top, padding_bottom)

    # Input tensor
    input_tensor = torch.randn(input_size)

    # ZeroPad2d module
    zero_pad_2d = torch.nn.ZeroPad2d(padding)

    # Padded output
    result = zero_pad_2d(input_tensor)
    return result
