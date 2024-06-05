
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MarginRankingLoss)
class TorchMarginRankingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_marginrankingloss_correctness(self):
        input1 = torch.randn(random.randint(1, 10))
        input2 = torch.randn(random.randint(1, 10))
        target = torch.randint(0, 2, (random.randint(1, 10),), dtype=torch.long)
        margin_ranking_loss = torch.nn.MarginRankingLoss()
        result = margin_ranking_loss(input1, input2, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_marginrankingloss_large_scale(self):
        input1 = torch.randn(random.randint(1000, 10000))
        input2 = torch.randn(random.randint(1000, 10000))
        target = torch.randint(0, 2, (random.randint(1000, 10000),), dtype=torch.long)
        margin_ranking_loss = torch.nn.MarginRankingLoss()
        result = margin_ranking_loss(input1, input2, target)
        return result

