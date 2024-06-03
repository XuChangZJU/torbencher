
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.gammainc)
class TorchSpecialGammaincTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gammainc_0d(self, input=None):
        if input is not None:
            result = torch.special.gammainc(input[0], input[1])
            return result
        a = torch.rand([])
        b = torch.rand([])
        result = torch.special.gammainc(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_gammainc_1d(self, input=None):
        if input is not None:
            result = torch.special.gammainc(input[0], input[1])
            return result
        a = torch.rand(5)
        b = torch.rand(5)
        result = torch.special.gammainc(a, b)
        return result

