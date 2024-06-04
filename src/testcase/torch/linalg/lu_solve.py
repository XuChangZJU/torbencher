
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lu_solve)
class TorchLinalgLuSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9")
    def test_lu_solve(self):
        
        a = torch.randn(3, 3)
        LU, pivots = torch.linalg.lu_factor(a)
        b = torch.randn(3, 1)
        result = torch.linalg.lu_solve(b, LU, pivots)
        return result
