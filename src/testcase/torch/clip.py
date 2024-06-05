
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clip)
class TorchClipTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clip_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        min = random.uniform(0.1, 10.0)
        max = random.uniform(min, min + 10.0)
        result = torch.clip(tensor, min=min, max=max)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clip_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        min = random.uniform(0.1, 10.0)
        max = random.uniform(min, min + 10.0)
        result = torch.clip(tensor, min=min, max=max)
        return result

