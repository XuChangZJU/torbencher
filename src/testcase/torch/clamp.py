
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clamp)
class TorchClampTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp_min(self, input=None):
        if input is not None:
            result = torch.clamp(input[0], min=input[1])
            return [result, input]
        a = torch.randn(4)
        result = torch.clamp(a, min=0)
        return [result, [a, 0]]

    @test_api_version.larger_than("1.1.3")
    def test_clamp_max(self, input=None):
        if input is not None:
            result = torch.clamp(input[0], max=input[1])
            return [result, input]
        a = torch.randn(4)
        result = torch.clamp(a, max=0)
        return [result, [a, 0]]

    @test_api_version.larger_than("1.1.3")
    def test_clamp(self, input=None):
        if input is not None:
            result = torch.clamp(input[0], min=input[1], max=input[2])
            return [result, input]
        a = torch.randn(4)
        result = torch.clamp(a, min=-0.5, max=0.5)
        return [result, [a, -0.5, 0.5]]

