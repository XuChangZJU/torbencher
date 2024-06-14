import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyConvTranspose2d)
class TorchNnLazyconvtranspose2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_conv_transpose2d_correctness(self):
    # Randomly generate parameters for LazyConvTranspose2d
    out_channels = random.randint(1, 10)  # Random number of output channels
    kernel_size = random.randint(1, 5)  # Random kernel size
    stride = random.randint(1, 3)  # Random stride
    padding = random.randint(0, 2)  # Random padding
    output_padding = random.randint(0, 2)  # Random output padding
    groups = random.randint(1, 3)  # Random number of groups
    dilation = random.randint(1, 2)  # Random dilation

    # Randomly generate input tensor dimensions
    batch_size = random.randint(1, 4)  # Random batch size
    in_channels = random.randint(1, 10)  # Random number of input channels
    height = random.randint(5, 10)  # Random height of the input tensor
    width = random.randint(5, 10)  # Random width of the input tensor

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Initialize LazyConvTranspose2d with random parameters
    lazy_conv_transpose2d = torch.nn.LazyConvTranspose2d(
        out_channels, kernel_size, stride, padding, output_padding, groups, True, dilation
    )

    # Apply the LazyConvTranspose2d to the input tensor
    result = lazy_conv_transpose2d(input_tensor)
    return result
