
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dot)
class TorchDotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dot(self, input=None):
        if input is not None:
            result = torch.dot(input[0], input[1])
            return result
        a = torch.randn(5)
        b = torch.randn(5)
        result = torch.dot(a, b)
        return result

