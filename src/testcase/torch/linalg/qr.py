
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.qr)
class TorchLinalgQrTestCase(TorBencherTestCaseBase):
    def test_qr_4d(self):
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.qr(a)
        return result

