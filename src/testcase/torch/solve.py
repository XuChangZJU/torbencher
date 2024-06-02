
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.solve)
class TorchSolveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_solve_4d(self, input=None):
        if input is not None:
            result = torch.solve(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        b = torch.randn(4, 2)
        result = torch.solve(b, a)
        return [result, [a, b]]

