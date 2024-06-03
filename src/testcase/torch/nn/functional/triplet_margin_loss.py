
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.triplet_margin_loss)
class TorchNNFunctionalTripletMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_triplet_margin_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.triplet_margin_loss(input[0], input[1], input[2], margin=input[3], p=input[4], eps=input[5], swap=input[6], size_average=input[7], reduce=input[8], reduction=input[9])
            return result
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        c = torch.randn(3, 2)
        d = 1.0
        e = 2.0
        f = 1e-06
        g = False
        h = True
        i = True
        j = 'mean'
        result = torch.nn.functional.triplet_margin_loss(a, b, c, margin=d, p=e, eps=f, swap=g, size_average=h, reduce=i, reduction=j)
        return result


