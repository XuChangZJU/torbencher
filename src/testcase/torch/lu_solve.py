
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lu_solve)
class TorchLuSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lu_solve_4d(self, input=None):
        if input is not None:
            result = torch.lu_solve(input[0], input[1], input[2])
            return result
        a = torch.randn(4, 4)
        b = torch.randn(4, 2)
        lu_data, pivots = torch.lu(a)
        result = torch.lu_solve(b, lu_data, pivots)
        return result

