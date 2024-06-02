
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.normal)
class TorchNormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_normal_mean(self, input=None):
        if input is not None:
            result = torch.normal(mean=input[0], std=input[1])
            return [result, input]
        a = torch.arange(1., 11.)
        b = torch.arange(1, 0, -0.1)
        result = torch.normal(mean = a, std = b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_normal(self, input=None):
        if input is not None:
            result = torch.normal(input[0], input[1])
            return [result, input]
        a = 1.
        b = 10.
        result = torch.normal(a, b, (4, 4))
        return [result, [a, b, (4, 4)]]

