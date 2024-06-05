
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randint)
class TorchRandintTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randint_correctness(self):
        high = random.randint(1, 10)
        size = (random.randint(1, 10),)
        result = torch.randint(0, high, size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_randint_large_scale(self):
        high = random.randint(1000, 10000)
        size = (random.randint(1000, 10000),)
        result = torch.randint(0, high, size)
        return result

