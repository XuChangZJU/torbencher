
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.binary_cross_entropy_with_logits)
class TorchNNFunctionalBinaryCrossEntropyWithLogitsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_with_logits_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.binary_cross_entropy_with_logits(input[0], input[1], weight=input[2], size_average=input[3], reduce=input[4], reduction=input[5], pos_weight=input[6])
            return [result, input]
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        c = None
        d = True
        e = True
        f = 'mean'
        g = None
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, b, weight=c, size_average=d, reduce=e, reduction=f, pos_weight=g)
        return [result, [a, b, c, d, e, f, g]]


