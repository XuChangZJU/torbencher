import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cholesky_solve)
class TorchCholeskySolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cholesky_solve_4d(self, input=None):
        if input is not None:
            result = torch.cholesky_solve(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        a = a @ a.t()  # Make a symmetric positive definite matrix
        b = torch.randn(4, 2)
        l = torch.cholesky(a)
        result = torch.cholesky_solve(b, l)
        return [result, [b, l]]

