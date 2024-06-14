import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool3d)
class TorchNnMaxunpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool3d_correctness(self):
    # Randomly generate dimensions for the input tensor
    N = random.randint(1, 4)  # Batch size
    C = random.randint(1, 4)  # Number of channels
    D_in = random.randint(5, 10)  # Depth of the input tensor
    H_in = random.randint(5, 10)  # Height of the input tensor
    W_in = random.randint(5, 10)  # Width of the input tensor

    # Randomly generate kernel size, stride, and padding
    kernel_size = random.randint(2, 4)
    stride = kernel_size  # To ensure valid unpooling
    padding = random.randint(0, 1)

    # Create random input tensor
    input_tensor = torch.randn(N, C, D_in, H_in, W_in)

    # Create MaxPool3d and MaxUnpool3d layers
    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    unpool = torch.nn.MaxUnpool3d(kernel_size, stride=stride, padding=padding)

    # Perform max pooling
    pooled_output, indices = pool(input_tensor)

    # Perform max unpooling
    unpooled_output = unpool(pooled_output, indices)

    return unpooled_output
