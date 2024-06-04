
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.poisson_nll_loss)
class TorchNNFunctionalPoissonNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_poisson_nll_loss_common(self):
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        c = True
        d = False
        e = True
        f = 1e-08
        g = True
        h = 'mean'
        result = torch.nn.functional.poisson_nll_loss(a, b, log_input=c, full=d, size_average=e, eps=f, reduce=g, reduction=h)
        return result


