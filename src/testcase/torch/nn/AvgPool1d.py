
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AvgPool1d)
class TorchAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avgpool1d_correctness(self):
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        input_size = random.randint(1, 10)
        kernel_size = random.randint(1, input_size)
        stride = random.randint(1, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(batch_size, in_channels, input_size)
        avg_pool = torch.nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
        result = avg_pool(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_avgpool1d_large_scale(self):
        batch_size = random.randint(1000, 10000)
        in_channels = random.randint(100, 1000)
        input_size = random.randint(1000, 10000)
        kernel_size = random.randint(100, input_size)
        stride = random.randint(10, kernel_size)
        padding = random.randint(0, kernel_size)
        input_tensor = torch.randn(batch_size, in_channels, input_size)
        avg_pool = torch.nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
        result = avg_pool(input_tensor)
        return result

