import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyConv2d)
class TorchNnLazyconv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazyconv2d_correctness(self):
    # Randomly generate parameters for LazyConv2d
    out_channels = random.randint(1, 10)  # Number of output channels
    kernel_size = random.randint(1, 5)  # Size of the convolving kernel
    stride = random.randint(1, 3)  # Stride of the convolution
    padding = random.randint(0, 2)  # Zero-padding added to both sides of the input
    dilation = random.randint(1, 3)  # Spacing between kernel elements
    groups = random.randint(1, 3)  # Number of blocked connections from input channels to output channels
    bias = random.choice([True, False])  # Whether to add a learnable bias to the output

    # Randomly generate input tensor dimensions
    batch_size = random.randint(1, 4)  # Batch size
    in_channels = random.randint(1, 10)  # Number of input channels
    height = random.randint(10, 20)  # Height of the input tensor
    width = random.randint(10, 20)  # Width of the input tensor

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Initialize LazyConv2d with random parameters
    lazy_conv2d = torch.nn.LazyConv2d(out_channels, kernel_size, stride, padding, dilation, groups, bias)

    # Apply LazyConv2d to the input tensor
    result = lazy_conv2d(input_tensor)
    return result
