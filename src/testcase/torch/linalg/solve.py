
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.solve)
class TorchLinalgSolveTestCase(TorBencherTestCaseBase):
    def test_solve_4d(self):
        a = torch.randn(2, 2, 3, 3)
        b = torch.randn(2, 2, 3, 1)
        result = torch.linalg.solve(a, b)
        return result

