
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.functional.jvp)
class TorchAutogradFunctionalJvpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.5.0")
    def test_jvp_correctness(self):
        dim = random.randint(1, 10)
        x = torch.randn(dim, requires_grad=True)
        func = lambda x: x + 1
        v = torch.randn(dim)
        result = torch.autograd.functional.jvp(func, x, v)
        return result

    @test_api_version.larger_than("1.5.0")
    def test_jvp_large_scale(self):
        dim = random.randint(1000, 10000)
        x = torch.randn(dim, requires_grad=True)
        func = lambda x: x + 1
        v = torch.randn(dim)
        result = torch.autograd.functional.jvp(func, x, v)
        return result

