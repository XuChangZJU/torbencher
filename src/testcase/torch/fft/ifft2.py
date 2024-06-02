
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.ifft2)
class TorchIfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_ifft2_4d(self, input=None):
        if input is not None:
            result = torch.fft.ifft2(input[0], input[1], input[2], input[3], input[4])
            return [result, input]
        a = torch.randn(4, 3, 8, 8, dtype=torch.cfloat)
        b = [2, 3]
        c = [-2, -1]
        d = "ortho"
        result = torch.fft.ifft2(a, b, c, d)
        return [result, [a, b, c, d]]

