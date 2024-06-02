
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.cond)
class TorchLinalgCondTestCase(TorBencherTestCaseBase):
    def test_cond_4d(self, input=None):
        if input is not None:
            result = torch.linalg.cond(input[0])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        result = torch.linalg.cond(a)
        return [result, [a]]

