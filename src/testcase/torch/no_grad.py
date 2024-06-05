
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.no_grad)
class TorchNoGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_no_grad_correctness(self):
        x = torch.randn(random.randint(1, 10), requires_grad=True)
        with torch.no_grad():
            y = x * 2
        result = y.requires_grad
        return result

    @test_api_version.larger_than("1.1.3")
    def test_no_grad_large_scale(self):
        x = torch.randn(random.randint(1000, 10000), requires_grad=True)
        with torch.no_grad():
            y = x * 2
        result = y.requires_grad
        return result

