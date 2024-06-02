
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mul)
class TorchMulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mul_number(self, input=None):
        if input is not None:
            result = torch.mul(input[0], input[1])
            return [result, input]
        a = torch.randn(4)
        result = torch.mul(a, 10)
        return [result, [a, 10]]

    @test_api_version.larger_than("1.1.3")
    def test_mul(self, input=None):
        if input is not None:
            result = torch.mul(input[0], input[1])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.mul(a, b)
        return [result, [a, b]]

