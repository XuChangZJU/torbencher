
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_unpool2d)
class MaxUnpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_unpool2d_correctness(self):
        batch_size = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)
        output_size = (random.randint(10, 20), random.randint(10, 20))
        input_data = torch.randn(batch_size, channel, height, width)
        indices = torch.randint(0, height * width, (batch_size, channel, height, width))
        result = torch.nn.functional.max_unpool2d(input_data, indices, kernel_size, stride, padding, output_size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max_unpool2d_large_scale(self):
        batch_size = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(1000, 2000)
        width = random.randint(1000, 2000)
        kernel_size = random.randint(10, 50)
        stride = random.randint(1, 10)
        padding = random.randint(0, 10)
        output_size = (random.randint(1000, 2000), random.randint(1000, 2000))
        input_data = torch.randn(batch_size, channel, height, width)
        indices = torch.randint(0, height * width, (batch_size, channel, height, width))
        result = torch.nn.functional.max_unpool2d(input_data, indices, kernel_size, stride, padding, output_size)
        return result

