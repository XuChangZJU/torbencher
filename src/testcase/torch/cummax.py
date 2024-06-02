
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cummax)
class TorchCummaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cummax(self, input=None):
        if input is not None:
            result = torch.cummax(input[0], input[1])
            return [result[0], input]
        a = torch.randn(10)
        result = torch.cummax(a, dim=0)
        return [result[0], [a, 0]]

