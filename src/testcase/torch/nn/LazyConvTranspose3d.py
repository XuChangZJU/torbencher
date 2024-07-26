import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.LazyConvTranspose3d)
class TorchNnLazyconvtranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_conv_transpose3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        depth = random.randint(5, 10)
        height = random.randint(5, 10)
        width = random.randint(5, 10)

        # Randomly generate parameters for LazyConvTranspose3d
        out_channels = random.randint(1, 4)
        kernel_size = random.randint(1, 4)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        groups = 1  # Default value
        bias = True  # Default value
        dilation = 1  # Default value
        output_padding = random.randint(0, stride - 1)  # 确保output_padding严格小于stride

        # Create a random input tensor
        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)

        # Initialize LazyConvTranspose3d with random parameters
        lazy_conv_transpose3d = torch.nn.LazyConvTranspose3d(out_channels, kernel_size, stride, padding, output_padding,
                                                             groups, bias, dilation)

        # Apply the LazyConvTranspose3d to the input tensor
        result = lazy_conv_transpose3d(input_tensor)

        return result
