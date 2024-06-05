
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.swapaxes)
class TorchSwapaxesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_swapaxes_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        axis1 = random.randint(0, 2)
        axis2 = random.randint(0, 2)
        result = torch.swapaxes(input, axis1, axis2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_swapaxes_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        axis1 = random.randint(0, 2)
        axis2 = random.randint(0, 2)
        result = torch.swapaxes(input, axis1, axis2)
        return result

