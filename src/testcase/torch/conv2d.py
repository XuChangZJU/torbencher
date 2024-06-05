
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.conv2d)
class TorchConv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv2d_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        dilation = random.randint(1, 10)
        groups = random.randint(1, 10)
        input = torch.randn(batch, channel, height, width)
        weight = torch.randn(channel, groups, kernel_size, kernel_size)
        bias = torch.randn(channel)
        result = torch.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv2d_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(100, 1000)
        width = random.randint(100, 1000)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        dilation = random.randint(1, 10)
        groups = random.randint(1, 10)
        input = torch.randn(batch, channel, height, width)
        weight = torch.randn(channel, groups, kernel_size, kernel_size)
        bias = torch.randn(channel)
        result = torch.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return result

