
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.signbit)
class TorchSignbitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_signbit(self, input=None):
        if input is not None:
            result = torch.signbit(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.signbit(a)
        return [result, [a]]


