
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arctan2)
class TorchArctan2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctan2(self, input=None):
        if input is not None:
            result = torch.arctan2(input[0], input[1])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.arctan2(a, b)
        return [result, [a, b]]

