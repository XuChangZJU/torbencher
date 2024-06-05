
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.zeros)
class TorchZerosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zeros_correctness(self):
        size = (random.randint(1, 10),)
        result = torch.zeros(size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_zeros_large_scale(self):
        size = (random.randint(1000, 10000),)
        result = torch.zeros(size)
        return result

