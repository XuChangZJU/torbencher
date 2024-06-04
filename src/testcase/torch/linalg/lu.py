
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lu)
class TorchLinalgLuTestCase(TorBencherTestCaseBase):
    def test_lu_4d(self):
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.lu(a)
        return result

