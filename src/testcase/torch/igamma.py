
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.igamma)
class TorchIgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_igamma_0d(self, input=None):
        if input is not None:
            result = torch.igamma(input[0], input[1])
            return [result, input]
        a = torch.randn(()) + 1
        b = torch.randn(()) + 1
        result = torch.igamma(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_igamma_1d(self, input=None):
        if input is not None:
            result = torch.igamma(input[0], input[1])
            return [result, input]
        a = torch.randn(4) + 1
        b = torch.randn(4) + 1
        result = torch.igamma(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_igamma_2d(self, input=None):
        if input is not None:
            result = torch.igamma(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4) + 1
        b = torch.randn(4, 4) + 1
        result = torch.igamma(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_igamma_3d(self, input=None):
        if input is not None:
            result = torch.igamma(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4, 4) + 1
        b = torch.randn(4, 4, 4) + 1
        result = torch.igamma(a, b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.1.3")
    def test_igamma_4d(self, input=None):
        if input is not None:
            result = torch.igamma(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4, 4, 4) + 1
        b = torch.randn(4, 4, 4, 4) + 1
        result = torch.igamma(a, b)
        return [result, [a, b]]


