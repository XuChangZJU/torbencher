
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.TripletMarginWithDistanceLoss)
class TorchNNTripletMarginWithDistanceLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_with_distance_loss(self, input=None):
        if input is not None:
            result = torch.nn.TripletMarginWithDistanceLoss()(input[0], input[1], input[2])
            return result
        anchor = torch.randn(100, 128)
        positive = torch.randn(100, 128)
        negative = torch.randn(100, 128)
        loss = torch.nn.TripletMarginWithDistanceLoss()
        result = loss(anchor, positive, negative)
        return result

