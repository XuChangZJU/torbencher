
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lgamma)
class TorchLgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lgamma_0d(self, input=None):
        if input is not None:
            result = torch.lgamma(input[0])
            return [result, input]
        a = torch.randn(())
        result = torch.lgamma(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_lgamma_1d(self, input=None):
        if input is not None:
            result = torch.lgamma(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.lgamma(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_lgamma_2d(self, input=None):
        if input is not None:
            result = torch.lgamma(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.lgamma(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_lgamma_3d(self, input=None):
        if input is not None:
            result = torch.lgamma(input[0])
            return [result, input]
        a = torch.randn(4, 4, 4)
        result = torch.lgamma(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_lgamma_4d(self, input=None):
        if input is not None:
            result = torch.lgamma(input[0])
            return [result, input]
        a = torch.randn(4, 4, 4, 4)
        result = torch.lgamma(a)
        return [result, [a]]

# torch.log
