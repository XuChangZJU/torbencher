
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lu_solve)
class TorchLinalgLuSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9")
    def test_lu_solve(self, input=None):
        if input is not None:
            result = torch.linalg.lu_solve(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(3, 3)
        LU, pivots = torch.linalg.lu_factor(a)
        b = torch.randn(3, 1)
        result = torch.linalg.lu_solve(b, LU, pivots)
        return [result, [b, LU, pivots]]
