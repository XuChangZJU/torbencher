
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.group_norm)
class GroupNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_group_norm_correctness(self):
        num_groups = random.randint(1, 10)
        num_channels = num_groups * random.randint(1, 10)
        input_data = torch.randn(10, num_channels, 10, 10)
        weight = torch.randn(num_channels)
        bias = torch.randn(num_channels)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.group_norm(input_data, num_groups, weight, bias, eps)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_group_norm_large_scale(self):
        num_groups = random.randint(100, 1000)
        num_channels = num_groups * random.randint(100, 1000)
        input_data = torch.randn(100, num_channels, 100, 100)
        weight = torch.randn(num_channels)
        bias = torch.randn(num_channels)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.group_norm(input_data, num_groups, weight, bias, eps)
        return result

