
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.range)
class TorchRangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_range_correctness(self):
        start = random.randint(1, 10)
        end = random.randint(1, 10)
        step = random.randint(1, 10)
        result = torch.range(start, end, step)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_range_large_scale(self):
        start = random.randint(1000, 10000)
        end = random.randint(1000, 10000)
        step = random.randint(1000, 10000)
        result = torch.range(start, end, step)
        return result

