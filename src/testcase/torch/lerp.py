
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lerp)
class TorchLerpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lerp(self, input=None):
        if input is not None:
            result = torch.lerp(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        c = 0.5
        result = torch.lerp(a, b, c)
        return [result, [a, b, c]]


