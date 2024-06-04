
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cummin)
class TorchCumminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cummin(self):
        a = torch.randn(10)
        result = torch.cummin(a, dim=0)
        return [result[0], [a, 0]]

