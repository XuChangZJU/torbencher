
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.slogdet)
class TorchLinalgSlogdetTestCase(TorBencherTestCaseBase):
    def test_slogdet_4d(self, input=None):
        if input is not None:
            result = torch.linalg.slogdet(input[0])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.slogdet(a)
        return [result, [a]]

