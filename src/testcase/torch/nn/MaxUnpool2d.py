import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool2d)
class TorchNnMaxunpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool2d_correctness(self):
    # Randomly generate dimensions for the input tensor
    N = random.randint(1, 4)  # Batch size
    C = random.randint(1, 4)  # Number of channels
    H_in = random.randint(4, 8)  # Height of the input tensor
    W_in = random.randint(4, 8)  # Width of the input tensor

    # Randomly generate kernel size and stride
    kernel_size = random.randint(2, 4)
    stride = kernel_size  # To ensure valid pooling and unpooling

    # Create random input tensor
    input_tensor = torch.randn(N, C, H_in, W_in)

    # Create MaxPool2d and MaxUnpool2d layers
    pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)

    # Perform max pooling
    pooled_output, indices = pool(input_tensor)

    # Perform max unpooling
    unpooled_output = unpool(pooled_output, indices)

    return unpooled_output
