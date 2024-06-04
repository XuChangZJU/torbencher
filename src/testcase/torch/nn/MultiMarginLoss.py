
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MultiMarginLoss)
class TorchNNMultiMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multimargin_loss(self):
        
        a = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        loss = torch.nn.MultiMarginLoss()
        result = loss(a, target)
        return result

