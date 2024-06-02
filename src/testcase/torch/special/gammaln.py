
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.gammaln)
class TorchSpecialGammalnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gammaln_0d(self, input=None):
        if input is not None:
            result = torch.special.gammaln(input[0])
            return [result, input]
        a = torch.rand([])
        result = torch.special.gammaln(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_gammaln_1d(self, input=None):
        if input is not None:
            result = torch.special.gammaln(input[0])
            return [result, input]
        a = torch.rand(5)
        result = torch.special.gammaln(a)
        return [result, [a]]

