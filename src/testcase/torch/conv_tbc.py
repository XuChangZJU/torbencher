
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.conv_tbc)
class TorchConvTbcTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_tbc_correctness(self):
        batch = random.randint(1, 10)
        input_channels = random.randint(1, 10)
        output_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        dilation = random.randint(1, 10)
        padding = random.randint(0, 10)
        input = torch.randn(batch, input_channels, kernel_size)
        weight = torch.randn(output_channels, input_channels, kernel_size)
        bias = torch.randn(output_channels)
        result = torch.conv_tbc(input, weight, bias, stride=stride, padding=padding, dilation=dilation)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv_tbc_large_scale(self):
        batch = random.randint(100, 1000)
        input_channels = random.randint(100, 1000)
        output_channels = random.randint(100, 1000)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        dilation = random.randint(1, 10)
        padding = random.randint(0, 10)
        input = torch.randn(batch, input_channels, kernel_size)
        weight = torch.randn(output_channels, input_channels, kernel_size)
        bias = torch.randn(output_channels)
        result = torch.conv_tbc(input, weight, bias, stride=stride, padding=padding, dilation=dilation)
        return result

