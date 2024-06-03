
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.eigvalsh)
class TorchLinalgEigvalshTestCase(TorBencherTestCaseBase):
    def test_eigvalsh_4d(self, input=None):
        if input is not None:
            result = torch.linalg.eigvalsh(input[0])
            return result
        a = torch.randn(2, 2, 3, 3)
        a = a + a.transpose(-1, -2)
        result = torch.linalg.eigvalsh(a)
        return result

