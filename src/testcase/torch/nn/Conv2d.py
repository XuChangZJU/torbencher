import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Conv2d)
class TorchNnConv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_conv2d_correctness(self):
        # Randomly generate parameters for Conv2d
        in_channels = random.randint(1, 10)  # Number of input channels
        out_channels = random.randint(1, 20)  # Number of output channels
        kernel_size = random.randint(1, 5)  # Size of the convolving kernel
        stride = random.randint(1, 3)  # Stride of the convolution
        padding = random.randint(0, 2)  # Padding added to all four sides of the input
        dilation = random.randint(1, 2)  # Spacing between kernel elements

        # Randomly generate input tensor size
        batch_size = random.randint(1, 5)  # Batch size
        height = random.randint(10, 50)  # Height of input planes in pixels
        width = random.randint(10, 50)  # Width of input planes in pixels
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        # Create Conv2d layer with generated parameters
        conv2d_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

        # Apply Conv2d layer to input tensor
        output_tensor = conv2d_layer(input_tensor)
        return output_tensor
