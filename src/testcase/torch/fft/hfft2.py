
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.hfft2)
class TorchHfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_hfft2_4d(self, input=None):
        if input is not None:
            result = torch.fft.hfft2(input[0], input[1], input[2], input[3], input[4])
            return result
        a = torch.randn(4, 3, 8, 8, dtype=torch.cfloat)
        b = [2, 3]
        c = [-2, -1]
        e = "ortho"
        result = torch.fft.hfft2(a, b, c, e)
        return result

