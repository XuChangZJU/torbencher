
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.UpsamplingBilinear2d)
class TorchNNUpsamplingBilinear2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsampling_bilinear2d(self):
        a = torch.randn(1, 2, 4, 4)
        upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        result = upsample(a)
        return result

