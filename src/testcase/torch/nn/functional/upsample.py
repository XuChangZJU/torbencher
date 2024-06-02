
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.upsample)
class TorchNNFunctionalUpsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample(self, input=None):
        if input is not None:
            result = torch.nn.functional.upsample(
                input[0], size=input[1], scale_factor=input[2], mode=input[3], align_corners=input[4]
            )
            return [result, input]
        a = torch.randn(2, 3, 8, 8)
        b = (16, 16)
        c = None
        d = 'nearest'
        e = None
        result = torch.nn.functional.upsample(a, size=b, scale_factor=c, mode=d, align_corners=e)
        return [result, [a, b, c, d, e]]


