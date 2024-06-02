
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.TripletMarginLoss)
class TorchNNTripletMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_loss(self, input=None):
        if input is not None:
            result = torch.nn.TripletMarginLoss(margin=input[0])(input[1], input[2], input[3])
            return [result, input]
        anchor = torch.randn(100, 128)
        positive = torch.randn(100, 128)
        negative = torch.randn(100, 128)
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        result = loss(anchor, positive, negative)
        return [result, [1.0, anchor, positive, negative]]

