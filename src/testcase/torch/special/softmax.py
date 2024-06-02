
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.softmax)
class TorchSpecialSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax_1d(self, input=None):
        if input is not None:
            result = torch.special.softmax(input[0], input[1])
            return [result, input]
        a = torch.randn(5)
        b = 0
        result = torch.special.softmax(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_softmax_2d(self, input=None):
        if input is not None:
            result = torch.special.softmax(input[0], input[1])
            return [result, input]
        a = torch.randn(2, 3)
        b = 1
        result = torch.special.softmax(a, b)
        return [result, [a, b]]

