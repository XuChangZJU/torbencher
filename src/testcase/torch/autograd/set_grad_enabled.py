
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.set_grad_enabled)
class TorchAutogradSetGradEnabledTestCase(TorBencherTestCaseBase):
    def test_set_grad_enabled(self, input=None):
        if input is not None:
            torch.autograd.set_grad_enabled(input[0])
            a = torch.tensor([1., 2., 3.], requires_grad=True)
            b = a + 2
            c = b.mean()
            return c.requires_grad]
        torch.autograd.set_grad_enabled(False)
        a = torch.tensor([1., 2., 3.], requires_grad=True)
        b = a + 2
        c = b.mean()
        return c.requires_grad


