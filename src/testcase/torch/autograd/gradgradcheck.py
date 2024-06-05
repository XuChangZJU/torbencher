
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.gradgradcheck)
class TorchAutogradGradgradcheckTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gradgradcheck_correctness(self):
        def fn(input):
            return input * 2

        input = torch.randn(random.randint(1, 10), requires_grad=True)
        result = torch.autograd.gradgradcheck(fn, input, eps=1e-6, atol=1e-4)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_gradgradcheck_large_scale(self):
        def fn(input):
            return input * 2

        input = torch.randn(random.randint(1000, 10000), requires_grad=True)
        result = torch.autograd.gradgradcheck(fn, input, eps=1e-6, atol=1e-4)
        return result


