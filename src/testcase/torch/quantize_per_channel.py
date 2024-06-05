
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.quantize_per_channel)
class TorchQuantizePerChannelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_quantize_per_channel_correctness(self):
        tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        scales = torch.rand(random.randint(1, 10))
        zero_points = torch.randint(0, 10, (random.randint(1, 10),))
        axis = random.randint(0, 1)
        result = torch.quantize_per_channel(tensor, scales, zero_points, axis)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_quantize_per_channel_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        scales = torch.rand(random.randint(1000, 10000))
        zero_points = torch.randint(0, 1000, (random.randint(1000, 10000),))
        axis = random.randint(0, 1)
        result = torch.quantize_per_channel(tensor, scales, zero_points, axis)
        return result

