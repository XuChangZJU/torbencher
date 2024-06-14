import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxPool1d)
class TorchNnMaxpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxpool1d_correctness(self):
    # Randomly generate dimensions for the input tensor
    batch_size = random.randint(1, 10)  # Random batch size
    num_channels = random.randint(1, 10)  # Random number of channels
    length = random.randint(10, 50)  # Random length of the input signal

    # Randomly generate parameters for MaxPool1d
    kernel_size = random.randint(1, 5)  # Random kernel size
    stride = random.randint(1, kernel_size)  # Random stride, must be <= kernel_size
    padding = random.randint(0, kernel_size // 2)  # Random padding, must be >= 0 and <= kernel_size / 2
    dilation = random.randint(1, 3)  # Random dilation, must be > 0

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, num_channels, length)

    # Apply MaxPool1d
    maxpool = torch.nn.MaxPool1d(kernel_size, stride, padding, dilation)
    output = maxpool(input_tensor)
    
    return output
