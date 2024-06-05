
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.full)
class TorchFullTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full_correctness(self):
        size = (random.randint(1, 10),)
        fill_value = random.uniform(0.1, 10.0)
        result = torch.full(size, fill_value)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_full_large_scale(self):
        size = (random.randint(1000, 10000),)
        fill_value = random.uniform(0.1, 10.0)
        result = torch.full(size, fill_value)
        return result

