
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.no_grad)
class TorchAutogradNoGradTestCase(TorBencherTestCaseBase):
    def test_no_grad(self, input=None):
        if input is not None:
            with torch.autograd.no_grad():
                a = torch.tensor([1., 2., 3.], requires_grad=True)
                b = a + 2
                c = b.mean()
            return [c.requires_grad, []]
        with torch.autograd.no_grad():
            a = torch.tensor([1., 2., 3.], requires_grad=True)
            b = a + 2
            c = b.mean()
        return [c.requires_grad, []]


