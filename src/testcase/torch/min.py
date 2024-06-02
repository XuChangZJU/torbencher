
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.min)
class TorchMinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_min_dim(self, input=None):
        if input is not None:
            result = torch.min(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.min(a, 1)
        return [result, [a, 1]]

    @test_api_version.larger_than("1.1.3")
    def test_min(self, input=None):
        if input is not None:
            result = torch.min(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.min(a)
        return [result, [a]]

