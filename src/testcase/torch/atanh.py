
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atanh)
class TorchAtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atanh_0d(self, input=None):
        if input is not None:
            result = torch.atanh(input[0])
            return [result, input]
        a = torch.randn(())
        result = torch.atanh(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_atanh_1d(self, input=None):
        if input is not None:
            result = torch.atanh(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.atanh(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_atanh_2d(self, input=None):
        if input is not None:
            result = torch.atanh(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.atanh(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_atanh_3d(self, input=None):
        if input is not None:
            result = torch.atanh(input[0])
            return [result, input]
        a = torch.randn(4, 4, 4)
        result = torch.atanh(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_atanh_4d(self, input=None):
        if input is not None:
            result = torch.atanh(input[0])
            return [result, input]
        a = torch.randn(4, 4, 4, 4)
        result = torch.atanh(a)
        return [result, [a]]

