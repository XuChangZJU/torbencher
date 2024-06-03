
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multilabel_soft_margin_loss)
class TorchNNFunctionalMultilabelSoftMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multilabel_soft_margin_loss(self, input=None):
        if input is not None:
            result = torch.nn.functional.multilabel_soft_margin_loss(
                input[0], input[1]
            )
            return result
        a = torch.randn(3, 2).sigmoid()
        b = torch.randn(3, 2)
        result = torch.nn.functional.multilabel_soft_margin_loss(a, b)
        return result


