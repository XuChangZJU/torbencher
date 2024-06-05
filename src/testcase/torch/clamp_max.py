
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clamp_max)
class TorchClampMaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp_max_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        max = random.uniform(0.1, 10.0)
        result = torch.clamp_max(tensor, max=max)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clamp_max_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        max = random.uniform(0.1, 10.0)
        result = torch.clamp_max(tensor, max=max)
        return result

