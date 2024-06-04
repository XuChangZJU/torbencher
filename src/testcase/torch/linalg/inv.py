
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.inv)
class TorchLinalgInvTestCase(TorBencherTestCaseBase):
    def test_inv_4d(self):
        a = torch.randn(2, 2, 3, 3)
        a = torch.matmul(a, a.transpose(-1, -2)) + 1e-05 * torch.eye(3, 3)
        result = torch.linalg.inv(a)
        return result

