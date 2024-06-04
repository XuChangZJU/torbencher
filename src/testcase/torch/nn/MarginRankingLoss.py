
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MarginRankingLoss)
class TorchNNMarginRankingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss(self):
        a = torch.randn(10, 5)
        b = torch.randn(10, 5)
        target = torch.randint(low=-1, high=2, size=(10,))
        loss = torch.nn.MarginRankingLoss()
        result = loss(a, b, target)
        return result

