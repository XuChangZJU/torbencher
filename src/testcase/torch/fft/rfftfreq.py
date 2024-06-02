
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.rfftfreq)
class TorchfftrfftfreqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_rfftfreq_1d(self, input=None):
        if input is not None:
            result = torch.fft.rfftfreq(input[0], d=input[1])
            return [result, input]
        a = 10
        b = 0.1
        result = torch.fft.rfftfreq(a, d=b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.7.0")
    def test_rfftfreq_1d_out(self, input=None):
        if input is not None:
            result = torch.fft.rfftfreq(input[0], d=input[1], out=input[2])
            return [result, input]
        a = 10
        b = 0.1
        c = torch.empty(6)
        result = torch.fft.rfftfreq(a, d=b, out=c)
        return [result, [a, b, c]]

