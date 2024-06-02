
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.interpolate)
class TorchNNFunctionalInterpolateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_interpolate_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.interpolate(input[0], size=input[1], scale_factor=input[2], mode=input[3], align_corners=input[4])
            return [result, input]
        a = torch.randn(1, 1, 3, 3)
        b = (6, 6)
        c = None
        d = 'nearest'
        e = None
        result = torch.nn.functional.interpolate(a, size=b, scale_factor=c, mode=d, align_corners=e)
        return [result, [a, b, c, d, e]]


