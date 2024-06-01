import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.triangular_solve)
class TorchTriangularSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triangular_solve_4d(self, input=None):
        if input is not None:
            result = torch.triangular_solve(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        a = torch.tril(a)  # Make lower triangular matrix
        b = torch.randn(4, 2)
        result = torch.triangular_solve(b, a)
        return [result, [b, a]]

