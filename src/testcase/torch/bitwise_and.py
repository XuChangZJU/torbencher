
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_and)
class TorchBitwiseAndTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_and_correctness(self):
        input = torch.randint(0, 10, (random.randint(1, 10),))
        other = torch.randint(0, 10, (random.randint(1, 10),))
        result = torch.bitwise_and(input, other)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_and_large_scale(self):
        input = torch.randint(0, 1000, (random.randint(1000, 10000),))
        other = torch.randint(0, 1000, (random.randint(1000, 10