
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.exp)
class TorchExpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exp(self, input=None):
        if input is not None:
            result = torch.exp(input[0])
            return result
        a = torch.randn(4)
        result = torch.exp(a)
        return result

