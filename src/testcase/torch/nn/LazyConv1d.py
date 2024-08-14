import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.LazyConv1d)
class TorchNnLazyconv1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_lazyconv1d_correctness(self):
        while True:
            out_channels = random.randint(1, 10)
            kernel_size = random.randint(1, min(5, 2 * random.randint(0, 2) + 1))
            stride = random.randint(1, 3)
            padding = random.randint(0, 2)
            dilation = random.randint(1, 3)
            groups = 1
            bias = random.choice([True, False])

            # Ensure kernel_size does not exceed input_length after padding, considering stride and dilation
            min_input_length_required = (kernel_size - 1) * dilation + 1  # Minimum length required to fit one kernel
            input_length_min = max(min_input_length_required, kernel_size - 2 * padding)  # Adjusted for padding

            # Calculate the effective kernel size considering dilation
            effective_kernel_size = (kernel_size - 1) * (dilation - 1) + 1

            # Ensure stride doesn't lead to a situation where no valid position exists post-convolution
            input_length_min = max(input_length_min, effective_kernel_size - stride + 1)

            # Generate input_length that can accommodate the conditions
            input_length = random.randint(input_length_min, 20)

            if input_length < input_length_min:
                continue

            break

        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 10)
        input_tensor = torch.randn(batch_size, in_channels, input_length)

        # Initialize LazyConv1d layer
        lazy_conv1d = torch.nn.LazyConv1d(out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Apply LazyConv1d to input tensor
        result = lazy_conv1d(input_tensor)

        # Optionally, add checks here to verify the output dimensions or other properties as needed

        return result
