
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.irfft2)
class TorchIrfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_irfft2_4d(self, input=None):
        if input is not None:
            result = torch.fft.irfft2(input[0], input[1], input[2], input[3], input[4], input[5])
            return result
        a = torch.randn(4, 3, 8, 8, dtype=torch.cfloat)
        b = [2, 3]
        c = [-2, -1]
        d = "ortho"
        result = torch.fft.irfft2(a, b, c, d)
        return result

