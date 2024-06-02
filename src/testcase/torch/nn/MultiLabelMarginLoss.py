
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MultiLabelMarginLoss)
class TorchNNMultiLabelMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multilabel_margin_loss(self, input=None):
        if input is not None:
            result = torch.nn.MultiLabelMarginLoss()(input[0], input[1])
            return [result, input]
        a = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, 5).random_(5)
        loss = torch.nn.MultiLabelMarginLoss()
        result = loss(a, target)
        return [result, [a, target]]

