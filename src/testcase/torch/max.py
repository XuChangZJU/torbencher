
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.max)
class TorchMaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_dim(self, input=None):
        if input is not None:
            result = torch.max(input[0], input[1])
            return result
        a = torch.randn(4, 4)
        result = torch.max(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_max(self, input=None):
        if input is not None:
            result = torch.max(input[0])
            return result
        a = torch.randn(4, 4)
        result = torch.max(a)
        return result

