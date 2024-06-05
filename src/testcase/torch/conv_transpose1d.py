
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.conv_transpose1d)
class TorchConvTranspose1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose1d_correctness(self):
        batch = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        output_padding = random.randint(0, 10)
        groups = random.randint(1, 10)
        dilation = random.randint(1, 10)
        input = torch.randn(batch, in_channels, random.randint(1, 10))
        weight = torch.randn(in_channels, groups, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.conv_transpose1d(input, weight, bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose1d_large_scale(self):
        batch = random.randint(100, 1000)
        in_channels = random.randint(100, 1000)
        out_channels = random.randint(100, 1000)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        output_padding = random.randint(0, 10)
        groups = random.randint(1, 10)
        dilation = random.randint(1, 10)
        input = torch.randn(batch, in_channels, random.randint(1, 10))
        weight = torch.randn(in_channels, groups, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.conv_transpose1d(input, weight, bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
        return result

