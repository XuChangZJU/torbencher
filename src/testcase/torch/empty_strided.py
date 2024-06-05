
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.empty_strided)
class TorchEmptyStridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_empty_strided_correctness(self):
        size = (random.randint(1, 10),)
        stride = (random.randint(1, 10),)
        result = torch.empty_strided(size, stride)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_empty_strided_large_scale(self):
        size = (random.randint(1000, 10000),)
        stride = (random.randint(1000, 10000),)
        result = torch.empty_strided(size, stride)
        return result

