
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.tensorsolve)
class TorchLinalgTensorsolveTestCase(TorBencherTestCaseBase):
    def test_tensorsolve_4d(self):
        
        a = torch.randn(2, 2, 2, 2)
        b = torch.randn(2, 2)
        result = torch.linalg.tensorsolve(a, b)
        return result
