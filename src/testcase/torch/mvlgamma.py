
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mvlgamma)
class TorchMvlgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mvlgamma(self):
        a = torch.randn(4).uniform_(1, 10)
        result = torch.mvlgamma(a, 2)
        return result


