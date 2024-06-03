
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool3d)
class TorchNNFunctionalAvgPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool3d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.avg_pool3d(input[0], input[1], stride=input[2], padding=input[3], ceil_mode=input[4], count_include_pad=input[5], divisor_override=input[6])
            return result
        a = torch.randn(1, 3, 8, 6, 4)
        b = (2, 2, 2)
        c = (2, 2, 2)
        d = 0
        e = False
        f = True
        g = 1
        result = torch.nn.functional.avg_pool3d(a, b, stride=c, padding=d, ceil_mode=e, count_include_pad=f, divisor_override=g)
        return result


