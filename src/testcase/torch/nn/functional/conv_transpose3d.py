
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv_transpose3d)
class TorchNNFunctionalConvTranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose3d_common(self):
        
        a = torch.randn(1, 3, 8, 8, 4)
        b = torch.randn(3, 3, 2, 2, 2)
        c = None
        d = 1
        e = 0
        f = 0
        g = 1
        h = 1
        result = torch.nn.functional.conv_transpose3d(a, b, bias=c, stride=d, padding=e, output_padding=f, groups=g, dilation=h)
        return result


