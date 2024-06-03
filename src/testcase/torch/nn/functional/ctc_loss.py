
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.ctc_loss)
class TorchNNFunctionalCTCLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.ctc_loss(input[0], input[1], input[2], input[3], blank=input[4], reduction=input[5], zero_infinity=input[6])
            return result
        a = torch.randn(30, 8, 20).log_softmax(2)
        b = torch.randint(1, 20, (8, 30), dtype=torch.long)
        c = torch.tensor([12, 15, 17, 20, 23, 26, 28, 30], dtype=torch.long)
        d = torch.tensor([30, 29, 28, 27, 26, 25, 24, 23], dtype=torch.long)
        e = 0
        f = 'mean'
        g = False
        result = torch.nn.functional.ctc_loss(a, b, c, d, blank=e, reduction=f, zero_infinity=g)
        return result


