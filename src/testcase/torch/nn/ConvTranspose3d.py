import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.ConvTranspose3d)
class TorchNnConvtranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_conv_transpose3d_correctness(self):
        # Randomly generate parameters for ConvTranspose3d
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        output_padding = random.randint(0, min(stride - 1, 2))
        dilation = random.randint(1, 2)
        groups = 1
        # Create ConvTranspose3d layer with random parameters
        conv_transpose3d = torch.nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation
        )

        # Randomly generate input tensor size
        batch_size = random.randint(1, 5)
        depth = random.randint(10, 20)
        height = random.randint(10, 20)
        width = random.randint(10, 20)

        # Create random input tensor
        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)

        # Apply ConvTranspose3d
        output_tensor = conv_transpose3d(input_tensor)

        return output_tensor
