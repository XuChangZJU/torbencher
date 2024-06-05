
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AvgPool2d)
class TorchAvgPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avgpool2d_correctness(self):
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        input_height = random.randint(1, 10)
        input_width = random.randint(1, 10)
        kernel_size = random.randint(1, min(input_height, input_width))
        stride = random.randint(1, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
        avg_pool = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        result = avg_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_avgpool2d_large_scale(self):
        batch_size = random.randint(1000, 10000)
        in_channels = random.randint(100, 1000)
        input_height = random.randint(1000, 10000)
        input_width = random.randint(1000, 10000)
        kernel_size = random.randint(100, min(input_height, input_width))
        stride = random.randint(10, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
        avg_pool = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        result = avg_pool(input_tensor)
        return result

