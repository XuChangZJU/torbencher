
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Conv2d)
class TorchConv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv2d_correctness(self):
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = (random.randint(1, 10), random.randint(1, 10))
        stride = (random.randint(1, kernel_size[0]), random.randint(1, kernel_size[1]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]))
        input_tensor = torch.randn(random.randint(1, 10), in_channels, random.randint(1, 10), random.randint(1, 10))
        conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        result = conv2d(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv2d_large_scale(self):
        in_channels = random.randint(100, 1000)
        out_channels = random.randint(100, 1000)
        kernel_size = (random.randint(100, 1000), random.randint(100, 1000))
        stride = (random.randint(10, kernel_size[0]), random.randint(10, kernel_size[1]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]))
        input_tensor = torch.randn(random.randint(1000, 10000), in_channels, random.randint(100, 1000), random.randint(100, 1000))
        conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        result = conv2d(input_tensor)
        return result

