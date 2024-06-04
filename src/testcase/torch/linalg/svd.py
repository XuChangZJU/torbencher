
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.svd)
class TorchLinalgSvdTestCase(TorBencherTestCaseBase):
    def test_svd_4d(self):
        
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.svd(a)
        return result

