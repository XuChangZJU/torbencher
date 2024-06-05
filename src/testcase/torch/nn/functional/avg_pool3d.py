
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool3d)
class AvgPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool3d_correctness(self):
        batch_size = random.randint(1, 10)
        channel = random.randint(1, 10)
        depth = random.randint(10, 20)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        ceil_mode = random.choice([True, False])
        count_include_pad = random.choice([True, False])
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.avg_pool3d(input_data, kernel_size, stride, padding, ceil_mode, count_include_pad)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_avg_pool3d_large_scale(self):
        batch_size = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        depth = random.randint(1000, 2000)
        height = random.randint(1000, 2000)
        width = random.randint(1000, 2000)
        kernel_size = random.randint(10, 50)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        ceil_mode = random.choice([True, False])
        count_include_pad = random.choice([True, False])
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.avg_pool3d(input_data, kernel_size, stride, padding, ceil_mode, count_include_pad)
        return result

