
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.margin_ranking_loss)
class TorchNNFunctionalMarginRankingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss_common(self):
        a = torch.randn(3)
        b = torch.randn(3)
        c = torch.randn(3)
        d = 0.6
        e = True
        f = True
        g = 'mean'
        result = torch.nn.functional.margin_ranking_loss(a, b, c, margin=d, size_average=e, reduce=f, reduction=g)
        return result


