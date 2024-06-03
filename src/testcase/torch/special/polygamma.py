
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.polygamma)
class TorchSpecialPolygammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_polygamma_0d(self, input=None):
        if input is not None:
            result = torch.special.polygamma(input[0], input[1])
            return result
        a = torch.randint(1, 5, ())
        b = torch.randn([])
        result = torch.special.polygamma(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_polygamma_1d(self, input=None):
        if input is not None:
            result = torch.special.polygamma(input[0], input[1])
            return result
        a = torch.randint(1, 5, ())
        b = torch.randn(5)
        result = torch.special.polygamma(a, b)
        return result

