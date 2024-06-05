
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.adaptive_max_pool1d)
class TorchAdaptiveMaxPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool1d_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        length = random.randint(1, 10)
        output_size = random.randint(1, 10)
        input = torch.randn(batch, channel, length)
        result = torch.adaptive_max_pool1d(input, output_size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool1d_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        length = random.randint(100, 1000)
        output_size = random.randint(1, 10)
        input = torch.randn(batch, channel, length)
        result = torch.adaptive_max_pool1d(input, output_size)
        return result

