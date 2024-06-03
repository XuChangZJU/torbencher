
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ger)
class TorchGerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ger_1d(self, input=None):
        if input is not None:
            result = torch.ger(input[0], input[1])
            return result
        a = torch.randn(5)
        b = torch.randn(4)
        result = torch.ger(a, b)
        return result

