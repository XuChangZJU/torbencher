import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.max_unpool1d)
class TorchNnFunctionalMaxUunpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_max_unpool1d_correctness(self):
        # Randomly generate the size of the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        length = random.randint(9, 11)  # Ensure length is within the valid range for output_size

        # Randomly generate kernel size, stride, and padding
        kernel_size = random.randint(2, 4)
        stride = random.randint(1, kernel_size)
        
        # Adjust padding to be at most half of kernel size
        padding = random.randint(0, kernel_size // 2)

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, channels, length)

        # Perform max pooling
        pooled_tensor, indices = torch.nn.functional.max_pool1d(input_tensor, kernel_size, stride, padding,
                                                                return_indices=True)

        # Calculate the output size for unpooling
        output_size = length  # Use the original length as the output size for max_unpool1d

        # Perform max unpooling
        result = torch.nn.functional.max_unpool1d(pooled_tensor, indices, kernel_size, stride, padding,
                                                  output_size=(output_size,))
        return result