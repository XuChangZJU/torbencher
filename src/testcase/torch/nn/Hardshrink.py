
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Hardshrink)
class TorchHardshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardshrink_correctness(self):
        lambd = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        hardshrink = torch.nn.Hardshrink(lambd=lambd)
        result = hardshrink(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hardshrink_large_scale(self):
        lambd = random.uniform(0.1, 10.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        hardshrink = torch.nn.Hardshrink(lambd=lambd)
        result = hardshrink(input_tensor)
        return result

