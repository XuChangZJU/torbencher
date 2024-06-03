
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hypot)
class TorchHypotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hypot_0d(self, input=None):
        if input is not None:
            result = torch.hypot(input[0], input[1])
            return result
        a = torch.randn(())
        b = torch.randn(())
        result = torch.hypot(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hypot_1d(self, input=None):
        if input is not None:
            result = torch.hypot(input[0], input[1])
            return result
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.hypot(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hypot_2d(self, input=None):
        if input is not None:
            result = torch.hypot(input[0], input[1])
            return result
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = torch.hypot(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hypot_3d(self, input=None):
        if input is not None:
            result = torch.hypot(input[0], input[1])
            return result
        a = torch.randn(4, 4, 4)
        b = torch.randn(4, 4, 4)
        result = torch.hypot(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_hypot_4d(self, input=None):
        if input is not None:
            result = torch.hypot(input[0], input[1])
            return result
        a = torch.randn(4, 4, 4, 4)
        b = torch.randn(4, 4, 4, 4)
        result = torch.hypot(a, b)
        return result


