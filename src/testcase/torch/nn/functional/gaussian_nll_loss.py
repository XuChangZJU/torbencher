
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.gaussian_nll_loss)
class TorchNNFunctionalGaussianNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gaussian_nll_loss(self):
        a = torch.randn(10, 3)
        b = torch.randn(10, 3)
        c = torch.randn(10, 3)
        result = torch.nn.functional.gaussian_nll_loss(a, b, c)
        return result


