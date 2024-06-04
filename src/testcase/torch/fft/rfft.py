
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.rfft)
class TorchRfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_rfft_4d(self):
        
        a = torch.randn(4, 3, 8, 8)
        b = None
        c = 0
        d = "ortho"
        result = torch.fft.rfft(a, b, c, d)
        return result

