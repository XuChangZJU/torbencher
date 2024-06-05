
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.avg_pool1d)
class TorchAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        length = random.randint(1, 10)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        input = torch.randn(batch, channel, length)
        result = torch.avg_pool1d(input, kernel_size=kernel_size, stride=stride, padding=padding)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        length = random.randint(100, 1000)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        input = torch.randn(batch, channel, length)
        result = torch.avg_pool1d(input, kernel_size=kernel_size, stride=stride, padding=padding)
        return result

