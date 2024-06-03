
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.norm)
class TorchLinalgNormKeepdimsTestCase(TorBencherTestCaseBase):
    def test_norm_4d_keepdims(self, input=None):
        if input is not None:
            result = torch.linalg.norm(input[0], keepdims=input[1])
            return result
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.norm(a, keepdims=True)
        return result

# torch.linalg.transpose
