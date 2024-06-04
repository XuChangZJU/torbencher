
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.backward)
class TorchAutogradBackwardTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_backward_0d(self):
        a = torch.randn(1, requires_grad=True)
        b = a ** 2
        c = b.mean()
        c.backward(retain_graph=True, create_graph=True)
        return c.grad

    @test_api_version.larger_than("1.1.3")
    def test_backward_1d(self):
        a = torch.randn(4, requires_grad=True)
        b = a ** 2
        c = b.mean()
        c.backward(retain_graph=True, create_graph=True)
        return c.grad


