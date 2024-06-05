
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv_transpose3d)
class ConvTranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose3d_correctness(self):
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        output_padding = random.randint(0, 2)
        groups = random.randint(1, in_channels)
        input_data = torch.randn(10, in_channels, 20, 20, 20)
        weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.nn.functional.conv_transpose3d(input_data, weight, bias, stride, padding, output_padding, groups)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose3d_large_scale(self):
        in_channels = random.randint(100, 1000)
        out_channels = random.randint(100, 1000)
        kernel_size = random.randint(10, 50)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        output_padding = random.randint(0, 10)
        groups = random.randint(1, in_channels)
        input_data = torch.randn(100, in_channels, 1000, 1000, 1000)
        weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.nn.functional.conv_transpose3d(input_data, weight, bias, stride, padding, output_padding, groups)
        return result

