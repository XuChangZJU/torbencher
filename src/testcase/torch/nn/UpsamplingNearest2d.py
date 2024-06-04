
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.UpsamplingNearest2d)
class TorchNNUpsamplingNearest2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsampling_nearest2d(self):
        
        a = torch.randn(1, 2, 4, 4)
        upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        result = upsample(a)
        return result

