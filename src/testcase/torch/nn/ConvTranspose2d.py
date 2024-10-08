import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest


@test_api(torch.nn.ConvTranspose2d)
class TorchNnConvtranspose2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
    def test_conv_transpose2d_correctness(self):
        # Randomly generate parameters for ConvTranspose2d
        in_channels = random.randint(2, 10)
        out_channels = random.randint(2, 10)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        output_padding = random.randint(0, min(stride - 1, 2))
        dilation = random.randint(1, 2)
        groups = 1
        # Create ConvTranspose2d layer with random parameters
        conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation
        )
        with torch.no_grad():
            conv_transpose2d.weight = torch.nn.Parameter(torch.randn(conv_transpose2d.weight.shape) * 0.01)
            if conv_transpose2d.bias is not None:
                conv_transpose2d.bias = torch.nn.Parameter(torch.randn(conv_transpose2d.bias.shape) * 0.01)
        # Randomly generate input tensor size
        batch_size = random.randint(1, 5)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        # Apply ConvTranspose2d to the input tensor
        output_tensor = conv_transpose2d(input_tensor)

        return output_tensor
