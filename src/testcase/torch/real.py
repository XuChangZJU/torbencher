
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.real)
class TorchRealTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_real(self, input=None):
        if input is not None:
            result = torch.real(input[0])
            return [result, input]
        a = torch.randn(4, dtype=torch.cfloat)
        result = torch.real(a)
        return [result, [a]]


