
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.grad)
class TorchAutogradGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grad_correctness(self):
        input = torch.randn(random.randint(1, 10), requires_grad=True)
        output = input * 2
        result = torch.autograd.grad(outputs=output, inputs=input, create_graph=random.choice([True, False]))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_grad_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), requires_grad=True)
        output = input * 2
        result = torch.autograd.grad(outputs=output, inputs=input, create_graph=random.choice([True, False]))
        return result


