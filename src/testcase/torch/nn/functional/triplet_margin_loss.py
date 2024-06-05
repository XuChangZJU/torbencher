
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.triplet_margin_loss)
class TripletMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_loss_correctness(self):
        anchor = torch.randn(10, 10)
        positive = torch.randn(10, 10)
        negative = torch.randn(10, 10)
        margin = random.uniform(0.0, 1.0)
        p = random.randint(1, 5)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin, p, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_loss_large_scale(self):
        anchor = torch.randn(1000, 1000)
        positive = torch.randn(1000, 1000)
        negative = torch.randn(1000, 1000)
        margin = random.uniform(0.0, 1.0)
        p = random.randint(1, 5)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin, p, reduction)
        return result

