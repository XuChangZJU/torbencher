
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.bessel_j1)
class TorchSpecialBesselJ1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bessel_j1_0d(self, input=None):
        if input is not None:
            result = torch.special.bessel_j1(input[0])
            return result
        a = torch.randn([])
        result = torch.special.bessel_j1(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bessel_j1_1d(self, input=None):
        if input is not None:
            result = torch.special.bessel_j1(input[0])
            return result
        a = torch.randn(5)
        result = torch.special.bessel_j1(a)
        return result

