
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky_solve)
class TorchCholeskySolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_2d_2d(self, input=None):
        if input is not None:
            result = torch.cholesky_solve(input[0], input[1])
            return [result, input]
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        b = torch.randn(3, 2)
        u = torch.cholesky(a)
        result = torch.cholesky_solve(b, u)
        return [result, [b, u]]

    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_2d_2d_upper(self, input=None):
        if input is not None:
            result = torch.cholesky_solve(input[0], input[1], upper=input[2])
            return [result, input]
        a = torch.randn(3, 3)
        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive-definite
        b = torch.randn(3, 2)
        u = torch.cholesky(a)
        upper = True
        result = torch.cholesky_solve(b, u, upper=upper)
        return [result, [b, u, upper]]


