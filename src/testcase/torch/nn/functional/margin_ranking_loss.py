
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.margin_ranking_loss)
class MarginRankingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss_correctness(self):
        input1 = torch.randn(10)
        input2 = torch.randn(10)
        target = torch.randint(0, 2, (10,)) * 2 - 1
        margin = random.uniform(0.0, 1.0)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.margin_ranking_loss(input1, input2, target, margin, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss_large_scale(self):
        input1 = torch.randn(1000)
        input2 = torch.randn(1000)
        target = torch.randint(0, 2, (1000,)) * 2 - 1
        margin = random.uniform(0.0, 1.0)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.margin_ranking_loss(input1, input2, target, margin, reduction)
        return result

