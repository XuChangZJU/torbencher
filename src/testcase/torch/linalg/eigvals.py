
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.eigvals)
class TorchLinalgEigvalsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_eigvals(self, input=None):
        if input is not None:
            result = torch.linalg.eigvals(input[0])
            return [result, input]
        a = torch.randn(3, 3)
        result = torch.linalg.eigvals(a)
        return [result, [a]]

