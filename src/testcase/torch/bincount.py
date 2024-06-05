
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bincount)
class TorchBincountTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bincount_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randint(0, 10, (dim,))
        result = torch.bincount(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bincount_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randint(0, 10, (dim,))
        result = torch.bincount(input)
        return result

