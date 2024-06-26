
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.exp2)
class TorchExp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exp2(self, input=None):
        if input is not None:
            result = torch.exp2(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.exp2(a)
        return [result, [a]]


