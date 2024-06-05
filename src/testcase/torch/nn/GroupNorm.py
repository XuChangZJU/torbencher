
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GroupNorm)
class TorchGroupNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_groupnorm_correctness(self):
        num_groups = random.randint(1, 10)
        num_channels = num_groups * random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), num_channels, random.randint(1, 10), random.randint(1, 10))
        group_norm = torch.nn.GroupNorm(num_groups, num_channels)
        result = group_norm(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_groupnorm_large_scale(self):
        num_groups = random.randint(100, 1000)
        num_channels = num_groups * random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), num_channels, random.randint(100, 1000), random.randint(100, 1000))
        group_norm = torch.nn.GroupNorm(num_groups, num_channels)
        result = group_norm(input_tensor)
        return result

