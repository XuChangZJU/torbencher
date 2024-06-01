import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mvlgamma)
class TorchMvlgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mvlgamma_4d(self, input=None):
        if input is not None:
            result = torch.mvlgamma(input[0], input[1])
            return [result, input]
        a = torch.randn(4).abs() + 1
        result = torch.mvlgamma(a, 2)
        return [result, [a, 2]]

