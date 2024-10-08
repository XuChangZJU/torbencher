import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.fractional_max_pool3d)
class TorchNnFunctionalFractionalUmaxUpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fractional_max_pool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 5)  # Batch size
        C = random.randint(1, 5)  # Number of channels
        T_in = random.randint(10, 20)  # Depth of the input tensor
        H_in = random.randint(10, 20)  # Height of the input tensor
        W_in = random.randint(10, 20)  # Width of the input tensor

        # Generate a random input tensor with the above dimensions
        input_tensor = torch.randn(N, C, T_in, H_in, W_in)

        # Randomly generate kernel size for the pooling operation
        kernel_size = random.randint(2, 5)

        # Ensure the kernel size is not larger than the input dimensions
        kernel_size = min(kernel_size, T_in, H_in, W_in)

        # Define constraints to ensure output size is valid
        max_T_out = (T_in - 1) // kernel_size + 1
        max_H_out = (H_in - 1) // kernel_size + 1
        max_W_out = (W_in - 1) // kernel_size + 1

        # Randomly generate output size for the pooling operation
        T_out = random.randint(1, max_T_out)
        H_out = random.randint(1, max_H_out)
        W_out = random.randint(1, max_W_out)
        output_size = (T_out, H_out, W_out)

        # Apply fractional max pooling
        result = torch.nn.functional.fractional_max_pool3d(input_tensor, kernel_size, output_size)

        return result
