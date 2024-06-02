
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv_transpose2d)
class TorchNNFunctionalConvTranspose2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose2d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.conv_transpose2d(input[0], input[1], bias=input[2], stride=input[3], padding=input[4], output_padding=input[5], groups=input[6], dilation=input[7])
            return [result, input]
        a = torch.randn(1, 3, 8, 8)
        b = torch.randn(3, 3, 2, 2)
        c = None
        d = 1
        e = 0
        f = 0
        g = 1
        h = 1
        result = torch.nn.functional.conv_transpose2d(a, b, bias=c, stride=d, padding=e, output_padding=f, groups=g, dilation=h)
        return [result, [a, b, c, d, e, f, g, h]]


