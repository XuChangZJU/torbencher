
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_max_pool3d)
class AdaptiveMaxPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool3d_correctness(self):
        batch_size = random.randint(1, 10)
        channel = random.randint(1, 10)
        depth = random.randint(10, 20)
        height = random.randint(10, 20)
        width = random.randint(10, 20)
        output_size = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.adaptive_max_pool3d(input_data, output_size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool3d_large_scale(self):
        batch_size = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        depth = random.randint(1000, 2000)
        height = random.randint(1000, 2000)
        width = random.randint(1000, 2000)
        output_size = (random.randint(100, 1000), random.randint(100, 1000), random.randint(100, 1000))
        input_data = torch.randn(batch_size, channel, depth, height, width)
        result = torch.nn.functional.adaptive_max_pool3d(input_data, output_size)
        return result

