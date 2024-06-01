import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Upsample)
class TorchNNUpsampleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsample(self, input=None):
        if input is not None:
            result = torch.nn.Upsample(input[0], mode=input[1], align_corners=input[2])(input[3])
            return [result, input]
        a = torch.randn(1, 2, 4, 4)
        upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        result = upsample(a)
        return [result, [None, 'bilinear', False, a]]

