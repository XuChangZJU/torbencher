
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sub)
class TorchSubTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sub_number(self, input=None):
        if input is not None:
            result = torch.sub(input[0], input[1])
            return result
        a = torch.randn(4)
        result = torch.sub(a, 10)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sub(self, input=None):
        if input is not None:
            result = torch.sub(input[0], input[1], alpha=input[2])
            return result
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.sub(a, b, alpha=10)
        return result

