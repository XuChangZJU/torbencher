
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Conv3d)
class TorchConv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv3d_correctness(self):
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        stride = (random.randint(1, kernel_size[0]), random.randint(1, kernel_size[1]), random.randint(1, kernel_size[2]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]), random.randint(0, kernel_size[2]))
        input_tensor = torch.randn(random.randint(1, 10), in_channels, random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        result = conv3d(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_conv3d_large_scale(self):
        in_channels = random.randint(100, 1000)
        out_channels = random.randint(100, 1000)
        kernel_size = (random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        stride = (random.randint(10, kernel_size[0]), random.randint(10, kernel_size[1]), random.randint(10, kernel_size[2]))
        padding = (random.randint(0, kernel_size[0]), random.randint(0, kernel_size[1]), random.randint(0, kernel_size[2]))
        input_tensor = torch.randn(random.randint(1000, 10000), in_channels, random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        result = conv3d(input_tensor)
        return result

