
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cumsum)
class TorchCumsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumsum(self, input=None):
        if input is not None:
            result = torch.cumsum(input[0], input[1])
            return [result, input]
        a = torch.randn(10)
        result = torch.cumsum(a, dim=0)
        return [result, [a, 0]]

