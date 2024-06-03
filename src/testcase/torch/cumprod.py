
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cumprod)
class TorchCumprodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumprod(self, input=None):
        if input is not None:
            result = torch.cumprod(input[0], input[1])
            return result
        a = torch.randn(10)
        result = torch.cumprod(a, dim=0)
        return result

