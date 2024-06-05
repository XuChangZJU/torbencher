
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.fractional_max_pool3d)
class FractionalMaxPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool3d_correctness(self):
        batch_size = random.randint(1, 10)
        channel = random.randint(1, 10)
        depth = random.randint(10, 20)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        kernel_size = (random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))
        output_size = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.fractional_max_pool3d(input_data, kernel_size, output_size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool3d_large_scale(self):
        batch_size = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        depth = random.randint(1000, 2000)
        height = random.randint(1000, 2000)
        width = random.randint(1000, 2000)
        kernel_size = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
        output_size = (random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.fractional_max_pool3d(input_data, kernel_size, output_size)
        return result

