
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multi_margin_loss)
class TorchNNFunctionalMultiMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multi_margin_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.multi_margin_loss(input[0], input[1], p=input[2], margin=input[3], weight=input[4], size_average=input[5], reduce=input[6], reduction=input[7])
            return [result, input]
        a = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
        b = torch.tensor([3])
        c = 1
        d = 1.0
        e = None
        f = True
        g = True
        h = 'mean'
        result = torch.nn.functional.multi_margin_loss(a, b, p=c, margin=d, weight=e, size_average=f, reduce=g, reduction=h)
        return [result, [a, b, c, d, e, f, g, h]]


