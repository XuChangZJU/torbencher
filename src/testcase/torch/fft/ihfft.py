
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.ihfft)
class TorchIhfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_ihfft_4d(self):
        a = torch.randn(4, 3, 8, 8)
        b = None
        c = 0
        d = "ortho"
        result = torch.fft.ihfft(a, b, c, d)
        return result

