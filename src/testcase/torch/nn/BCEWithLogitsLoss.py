
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BCEWithLogitsLoss)
class TorchNNBCEWithLogitsLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bce_with_logits_loss(self):
        a = torch.randn(3, requires_grad=True)
        b = torch.empty(3).random_(2)
        loss = torch.nn.BCEWithLogitsLoss()
        result = loss(a, b)
        return result

