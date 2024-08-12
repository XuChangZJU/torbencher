import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.MaxUnpool1d)
class TorchNnMaxunpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_maxunpool1d_correctness(self):
        # Randomly generate input tensor dimensions
        N = random.randint(1, 3)  # Batch size
        C = random.randint(1, 3)  # Number of channels

        # Define a reasonable range for H_in to avoid too many iterations in the loop later
        H_in_options = list(
            range(8, 13))  # Height options that can work with kernel_size 2-4 and stride equivalent to kernel_size
        H_in = random.choice(H_in_options)

        # Randomly generate parameters for MaxPool1d and MaxUnpool1d ensuring they are compatible with H_in
        kernel_size_options = [k for k in [2, 3, 4] if
                               (H_in - k) % k != 0 or H_in < k]  # Exclude sizes that would result in no pooling
        kernel_size = random.choice(
            kernel_size_options) if kernel_size_options else 2  # Default to 2 if others cause issues
        stride = kernel_size  # To ensure valid pooling and unpooling

        # Find a valid padding value directly without a loop to prevent infinite looping
        possible_paddings = [p for p in range(0, 2) if
                             (H_in - kernel_size + 2 * p) % stride == 0 and (H_in - kernel_size + 2 * p) / stride >= 1]
        padding = random.choice(possible_paddings) if possible_paddings else 0

        # Create random input tensor
        input_tensor = torch.randn(N, C, H_in)

        # Initialize MaxPool1d and MaxUnpool1d layers with the determined padding
        pool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True)
        unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)

        # Perform max pooling
        pooled_output, indices = pool(input_tensor)

        # Perform max unpooling
        unpooled_output = unpool(pooled_output, indices)

        return unpooled_output
