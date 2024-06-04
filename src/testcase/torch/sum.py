
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sum)
class TorchSumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sum_dim(self):
        a = torch.randn(4, 4)
        result = torch.sum(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sum(self):
        a = torch.randn(4, 4)
        result = torch.sum(a)
        return result

