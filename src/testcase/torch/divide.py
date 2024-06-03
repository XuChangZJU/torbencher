
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.divide)
class TorchDivideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_divide_0d(self, input=None):
        if input is not None:
            result = torch.divide(input[0], input[1])
            return result
        a = torch.randn(())
        b = torch.randn(())
        result = torch.divide(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_divide_1d(self, input=None):
        if input is not None:
            result = torch.divide(input[0], input[1])
            return result
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.divide(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_divide_2d(self, input=None):
        if input is not None:
            result = torch.divide(input[0], input[1])
            return result
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.divide(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_divide_3d(self, input=None):
        if input is not None:
            result = torch.divide(input[0], input[1])
            return result
        a = torch.randn(4, 4, 4)
        b = torch.randn(4, 4, 4)
        result = torch.divide(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_divide_4d(self, input=None):
        if input is not None:
            result = torch.divide(input[0], input[1])
            return result
        a = torch.randn(4, 4, 4, 4)
        b = torch.randn(4, 4, 4, 4)
        result = torch.divide(a, b)
        return result
