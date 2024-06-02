
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.backward)
class TorchAutogradBackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backward_0d(self, input=None):
        if input is not None:
            input[0].backward(gradient=input[1], retain_graph=input[2], create_graph=input[3])
            return [input[0].grad, input]
        a = torch.randn(1, requires_grad=True)
        b = a ** 2
        c = b.mean()
        c.backward(retain_graph=True, create_graph=True)
        return [c.grad, [c, None, True, True]]

    @test_api_version.larger_than("1.1.3")
    def test_backward_1d(self, input=None):
        if input is not None:
            input[0].backward(gradient=input[1], retain_graph=input[2], create_graph=input[3])
            return [input[0].grad, input]
        a = torch.randn(4, requires_grad=True)
        b = a ** 2
        c = b.mean()
        c.backward(retain_graph=True, create_graph=True)
        return [c.grad, [c, None, True, True]]


