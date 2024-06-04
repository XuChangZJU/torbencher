
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.ldl_solve)
class TorchLinalgLdlSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_ldl_solve(self):
        
        a = torch.randn(3, 3)
        a = (a + a.t()) / 2  # make symmetric
        LD, pivots = torch.linalg.ldl_factor(a, hermitian=True)
        b = torch.randn(3, 1)
        result = torch.linalg.ldl_solve(LD, pivots, b, hermitian=True)
        return result

