
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv2d)
class Conv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv2d_correctness(self):
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        dilation = random.randint(1, 2)
        groups = random.randint(1, in_channels)
        input_data = torch.randn(10, in_channels, 20, 20)
        weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.nn.functional.conv2d(input_data, weight, bias, stride, padding, dilation, groups)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv2d_large_scale(self):
        in_channels = random.randint(100, 1000)
        out_channels = random.randint(100, 1000)
        kernel_size = random.randint(10, 50)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        dilation = random.randint(1, 5)
        groups = random.randint(1, in_channels)
        input_data = torch.randn(100, in_channels, 1000, 1000)
        weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        bias = torch.randn(out_channels)
        result = torch.nn.functional.conv2d(input_data, weight, bias, stride, padding, dilation, groups)
        return result

