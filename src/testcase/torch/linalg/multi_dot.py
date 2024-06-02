
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.multi_dot)
class TorchLinalgMultiDotTestCase(TorBencherTestCaseBase):
    def test_multi_dot_4d(self, input=None):
        if input is not None:
            result = torch.linalg.multi_dot(input[0])
            return [result, input]
        a = torch.randn(2, 2, 3, 4)
        b = torch.randn(2, 2, 4, 5)
        c = torch.randn(2, 2, 5, 3)
        result = torch.linalg.multi_dot([a, b, c])
        return [result, [[a, b, c]]]

