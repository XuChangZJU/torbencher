
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.rfftn)
class TorchRfftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_rfftn_5d(self, input=None):
        if input is not None:
            result = torch.fft.rfftn(input[0], input[1], input[2], input[3], input[4])
            return [result, input]
        a = torch.randn(4, 3, 4, 5, 6)
        b = [2, 3, 5]
        c = [-3, -2, -1]
        e = "ortho"
        result = torch.fft.rfftn(a, b, c, e)
        return [result, [a, b, c, e]]

