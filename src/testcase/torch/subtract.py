
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.subtract)
class TorchSubtractTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_subtract(self, input=None):
        if input is not None:
            result = torch.subtract(input[0], input[1])
            return result
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.subtract(a, b)
        return result


