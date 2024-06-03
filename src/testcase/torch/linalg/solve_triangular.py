
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.solve_triangular)
class TorchLinalgSolveTriangularTestCase(TorBencherTestCaseBase):
    def test_solve_triangular_4d(self, input=None):
        if input is not None:
            result = torch.linalg.solve_triangular(input[0], input[1], upper=input[2])
            return result
        a = torch.randn(2, 2, 3, 3).triu()
        b = torch.randn(2, 2, 3, 1)
        result = torch.linalg.solve_triangular(a, b, upper=True)
        return result
