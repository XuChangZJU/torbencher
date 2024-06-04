
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.sinc)
class TorchSpecialSincTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_sinc_0d(self):
        a = torch.randn([])
        result = torch.special.sinc(a)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_sinc_1d(self):
        a = torch.randn(5)
        result = torch.special.sinc(a)
        return result



