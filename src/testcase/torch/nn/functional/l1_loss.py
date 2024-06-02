
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.l1_loss)
class TorchNNFunctionalL1LossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_l1_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.l1_loss(input[0], input[1], size_average=input[2], reduce=input[3], reduction=input[4])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        c = True
        d = True
        e = 'mean'
        result = torch.nn.functional.l1_loss(a, b, size_average=c, reduce=d, reduction=e)
        return [result, [a, b, c, d, e]]


