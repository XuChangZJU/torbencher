
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.enable_grad)
class TorchAutogradEnableGradTestCase(TorBencherTestCaseBase):
    def test_enable_grad(self, input=None):
        if input is not None:
            result = torch.autograd.enable_grad()
            return result
        result = torch.autograd.enable_grad()
        return result



