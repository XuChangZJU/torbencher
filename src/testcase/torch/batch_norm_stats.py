
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.batch_norm_stats)
class TorchBatchNormStatsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_stats_correctness(self):
        batch = random.randint(1, 10)
        channel = random.randint(1, 10)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        input = torch.randn(batch, channel, height, width)
        result = torch.batch_norm_stats(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_batch_norm_stats_large_scale(self):
        batch = random.randint(100, 1000)
        channel = random.randint(100, 1000)
        height = random.randint(100, 1000)
        width = random.randint(100, 1000)
        input = torch.randn(batch, channel, height, width)
        result = torch.batch_norm_stats(input)
        return result

