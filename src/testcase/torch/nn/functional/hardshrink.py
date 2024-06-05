
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardshrink)
class HardShrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardshrink_correctness(self):
        input_data = torch.randn(10, 10)
        lambd = random.uniform(0.0, 1.0)
        result = torch.nn.functional.hardshrink(input_data, lambd)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hardshrink_large_scale(self):
        input_data = torch.randn(1000, 1000)
        lambd = random.uniform(0.0, 1.0)
        result = torch.nn.functional.hardshrink(input_data, lambd)
        return result

