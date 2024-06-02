
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.nll_loss)
class TorchNNFunctionalNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nll_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.nll_loss(input[0], input[1], weight=input[2], size_average=input[3], ignore_index=input[4], reduce=input[5], reduction=input[6])
            return [result, input]
        a = torch.randn(3, 5)
        b = torch.tensor([1, 0, 4])
        c = None
        d = True
        e = -100
        f = True
        g = 'mean'
        result = torch.nn.functional.nll_loss(a, b, weight=c, size_average=d, ignore_index=e, reduce=f, reduction=g)
        return [result, [a, b, c, d, e, f, g]]


