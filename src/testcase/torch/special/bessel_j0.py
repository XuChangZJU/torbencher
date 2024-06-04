
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.bessel_j0)
class TorchSpecialBesselJ0TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bessel_j0_0d(self):
        a = torch.randn([])
        result = torch.special.bessel_j0(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bessel_j0_1d(self):
        a = torch.randn(5)
        result = torch.special.bessel_j0(a)
        return result

