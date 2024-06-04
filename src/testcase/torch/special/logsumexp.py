
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.logsumexp)
class TorchSpecialLogSumExpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_logsumexp_1d(self):
        a = torch.randn(5)
        b = 0
        result = torch.special.logsumexp(a, b)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_logsumexp_2d_dim_0(self):
        a = torch.randn(2, 3)
        b = 0
        result = torch.special.logsumexp(a, b)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_logsumexp_2d_dim_1(self):
        a = torch.randn(2, 3)
        b = 1
        result = torch.special.logsumexp(a, b)
        return result


