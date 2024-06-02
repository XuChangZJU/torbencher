
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.UpsamplingNearest2d)
class TorchNNUpsamplingNearest2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsampling_nearest2d(self, input=None):
        if input is not None:
            result = torch.nn.UpsamplingNearest2d(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 2, 4, 4)
        upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        result = upsample(a)
        return [result, [None, a]]

