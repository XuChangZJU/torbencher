import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.lp_pool1d)
class TorchNnFunctionalLpUpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_lp_pool1d_correctness(self):
        # Randomly generate the number of input planes
        num_input_planes = random.randint(1, 4)

        # Randomly generate the length of the input signal
        signal_length = random.randint(5, 10)

        # Randomly generate the power parameter p
        p = 1

        # Randomly generate the kernel size
        kernel_size = random.randint(2, 4)

        # Randomly generate the stride
        stride = random.randint(1, kernel_size)

        # Generate random input tensor with the shape (batch_size, num_input_planes, signal_length)
        batch_size = random.randint(1, 3)
        input_tensor = torch.randn(batch_size, num_input_planes, signal_length)

        # Apply lp_pool1d
        result = torch.nn.functional.lp_pool1d(input_tensor, p, kernel_size, stride)
        return result
