
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.poisson_nll_loss)
class TorchNNFunctionalPoissonNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_poisson_nll_loss_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.poisson_nll_loss(input[0], input[1], log_input=input[2], full=input[3], size_average=input[4], eps=input[5], reduce=input[6], reduction=input[7])
            return [result, input]
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        c = True
        d = False
        e = True
        f = 1e-08
        g = True
        h = 'mean'
        result = torch.nn.functional.poisson_nll_loss(a, b, log_input=c, full=d, size_average=e, eps=f, reduce=g, reduction=h)
        return [result, [a, b, c, d, e, f, g, h]]


