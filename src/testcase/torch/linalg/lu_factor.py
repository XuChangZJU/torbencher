
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lu_factor)
class TorchLinalgLuFactorTestCase(TorBencherTestCaseBase):
    def test_lu_factor_4d(self, input=None):
        if input is not None:
            result = torch.linalg.lu_factor(input[0])
            return result
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.lu_factor(a)
        return result

