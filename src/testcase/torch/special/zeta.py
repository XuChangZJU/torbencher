
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.zeta)
class TorchSpecialZetaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zeta_0d(self, input=None):
        if input is not None:
            result = torch.special.zeta(input[0], input[1])
            return result
        a = torch.rand([]) + 1
        b = torch.rand([])
        result = torch.special.zeta(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_zeta_1d(self, input=None):
        if input is not None:
            result = torch.special.zeta(input[0], input[1])
            return result
        a = torch.rand(5) + 1
        b = torch.rand(5)
        result = torch.special.zeta(a, b)
        return result

