
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv2d)
class TorchNNFunctionalConv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv2d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.conv2d(input[0], input[1], bias=input[2], stride=input[3], padding=input[4], dilation=input[5], groups=input[6])
            return [result, input]
        a = torch.randn(1, 3, 8, 8)
        b = torch.randn(3, 3, 2, 2)
        c = None
        d = 1
        e = 0
        f = 1
        g = 1
        result = torch.nn.functional.conv2d(a, b, bias=c, stride=d, padding=e, dilation=f, groups=g)
        return [result, [a, b, c, d, e, f, g]]


