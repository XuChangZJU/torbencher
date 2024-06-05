
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clamp_min)
class TorchClampMinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp_min_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        min = random.uniform(0.1, 10.0)
        result = torch.clamp_min(tensor, min=min)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clamp_min_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        min = random.uniform(0.1, 10.0)
        result = torch.clamp_min(tensor, min=min)
        return result

