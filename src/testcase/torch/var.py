
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.var)
class TorchVarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_var_dim(self, input=None):
        if input is not None:
            result = torch.var(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.var(a, 1)
        return [result, [a, 1]]

    @test_api_version.larger_than("1.1.3")
    def test_var(self, input=None):
        if input is not None:
            result = torch.var(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.var(a)
        return [result, [a]]

