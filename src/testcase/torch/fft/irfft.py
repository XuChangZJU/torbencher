
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.irfft)
class TorchIrfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_irfft_4d(self, input=None):
        if input is not None:
            result = torch.fft.irfft(input[0], input[1], input[2], input[3], input[4], input[5])
            return [result, input]
        a = torch.randn(4, 3, 8, 8, dtype=torch.cfloat)
        b = None
        c = None
        e = "ortho"
        result = torch.fft.irfft(a, b, c, e)
        return [result, [a, b, c, e]]

