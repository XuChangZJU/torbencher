
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.eigh)
class TorchLinalgEighTestCase(TorBencherTestCaseBase):
    def test_eigh_4d(self, input=None):
        if input is not None:
            result = torch.linalg.eigh(input[0])
            return result
        a = torch.randn(2, 2, 3, 3)
        a = a + a.transpose(-1, -2)
        result = torch.linalg.eigh(a)
        return result

