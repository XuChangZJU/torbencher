import random
from math import ceil

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.LazyConv3d)
class TorchNnLazyconv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_lazyconv3d_correctness(self):
        # Randomly generate parameters for LazyConv3d
        out_channels = random.randint(1, 10)
        kernel_size = (
        random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))  # Kernel size in all three dimensions
        stride = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))  # Stride for each dimension
        padding = (random.randint(0, 2), random.randint(0, 2), random.randint(0, 2))  # Padding for each side
        dilation = (
        random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))  # Dilation factor for each dimension

        # Ensure the input dimensions are compatible with the convolution settings
        while True:
            batch_size = random.randint(1, 4)
            in_channels = random.randint(1, 10)
            # Calculate minimum dimensions considering padding, stride, and dilation for each dimension
            min_depth = max(ceil((kernel_size[0] - 1) * dilation[0] + 1), kernel_size[0] - padding[0])
            min_height = max(ceil((kernel_size[1] - 1) * dilation[1] + 1), kernel_size[1] - padding[1])
            min_width = max(ceil((kernel_size[2] - 1) * dilation[2] + 1), kernel_size[2] - padding[2])

            depth = random.randint(max(min_depth, 10), 20)
            height = random.randint(max(min_height, 10), 20)
            width = random.randint(max(min_width, 10), 20)

            # Check if at least one valid position exists after convolution in all three dimensions
            if ((depth + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) < 1 or \
                    ((height + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) < 1 or \
                    ((width + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) < 1:
                continue

            break

        # Create a random input tensor with the validated dimensions
        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)

        # Initialize LazyConv3d with the generated parameters
        lazy_conv3d = torch.nn.LazyConv3d(out_channels, kernel_size, stride, padding, dilation)

        # Apply LazyConv3d to the input tensor
        result = lazy_conv3d(input_tensor)

        return result
