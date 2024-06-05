
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.triplet_margin_with_distance_loss)
class TripletMarginWithDistanceLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_with_distance_loss_correctness(self):
        anchor = torch.randn(10, 10)
        positive = torch.randn(10, 10)
        negative = torch.randn(10, 10)
        margin = random.uniform(0.0, 1.0)
        distance_function = torch.nn.functional.pairwise_distance
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, margin, distance_function, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_with_distance_loss_large_scale(self):
        anchor = torch.randn(1000, 1000)
        positive = torch.randn(1000, 1000)
        negative = torch.randn(1000, 1000)
        margin = random.uniform(0.0, 1.0)
        distance_function = torch.nn.functional.pairwise_distance
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, margin, distance_function, reduction)
        return result

