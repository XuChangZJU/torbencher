
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.as_strided)
class TorchAsStridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_as_strided_correctness(self):
        input = torch.randn(random.randint(1, 10))
        size = (random.randint(1, 10),)
        stride = (random.randint(1, 10),)
        result = torch.as_strided(input, size, stride)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_as_strided_large_scale(self):
        input = torch.randn(random.randint(1000, 10000))
        size = (random.randint(1000, 10000),)
        stride = (random.randint(1000, 10000),)
        result = torch.as_strided(input, size, stride)
        return result

