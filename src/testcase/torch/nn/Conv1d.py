import random
import random
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.Conv1d)
class TorchNnConv1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_Conv1d_correctness(self):
        # Random input size (N, C_in, L_in)
        batch_size = random.randint(1, 5)  # Reduced batch size for simplicity
        in_channels = random.randint(1, 10)  # Reduced range for in_channels
        in_length = random.randint(1, 20)  # Reduced range for input length
        input_size = [batch_size, in_channels, in_length]

        # Random Conv1d parameters
        out_channels = random.randint(1, 10)  # Reduced range for out_channels
        kernel_size = random.randint(1, in_length)  # Ensure kernel_size <= in_length
        stride = random.randint(1, max(1, (in_length - kernel_size) // 1 + 1))  # Ensure stride is valid

        # Create Conv1d module and random input tensor
        conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        input_tensor = torch.randn(input_size)

        # Perform convolution
        output_tensor = conv1d(input_tensor)

        return output_tensor

