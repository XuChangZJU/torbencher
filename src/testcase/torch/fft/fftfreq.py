
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.fftfreq)
class TorchfftfftfreqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_fftfreq_1d(self, input=None):
        if input is not None:
            result = torch.fft.fftfreq(input[0], d=input[1])
            return [result, input]
        a = 10
        b = 0.1
        result = torch.fft.fftfreq(a, d=b)
        return [result, [a, b]]

    @test_api_version.larger_than("1.7.0")
    def test_fftfreq_1d_out(self, input=None):
        if input is not None:
            result = torch.fft.fftfreq(input[0], d=input[1], out=input[2])
            return [result, input]
        a = 10
        b = 0.1
        c = torch.empty(10)
        result = torch.fft.fftfreq(a, d=b, out=c)
        return [result, [a, b, c]]

