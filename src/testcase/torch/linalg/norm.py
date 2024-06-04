
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.norm)
class TorchLinalgNormKeepdimsTestCase(TorBencherTestCaseBase):
    def test_norm_4d_keepdims(self):
        
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.norm(a, keepdims=True)
        return result

# torch.linalg.transpose
