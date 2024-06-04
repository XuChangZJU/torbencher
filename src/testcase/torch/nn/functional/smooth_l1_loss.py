
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.smooth_l1_loss)
class TorchNNFunctionalSmoothL1LossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss_common(self):
        a = torch.randn(4)
        b = torch.randn(4)
        c = True
        d = True
        e = 'mean'
        result = torch.nn.functional.smooth_l1_loss(a, b, size_average=c, reduce=d, reduction=e)
        return result


