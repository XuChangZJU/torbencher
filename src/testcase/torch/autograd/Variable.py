
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.Variable)
class TorchAutogradVariableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_variable_correctness(self):
        data = torch.randn(random.randint(1, 10))
        requires_grad = random.choice([True, False])
        result = torch.autograd.Variable(data, requires_grad=requires_grad)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_variable_large_scale(self):
        data = torch.randn(random.randint(1000, 10000))
        requires_grad = random.choice([True, False])
        result = torch.autograd.Variable(data, requires_grad=requires_grad)
        return result


