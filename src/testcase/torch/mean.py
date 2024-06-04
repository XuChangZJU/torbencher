
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mean)
class TorchMeanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mean_dim(self):
        a = torch.randn(4, 4)
        result = torch.mean(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_mean(self):
        a = torch.randn(4, 4)
        result = torch.mean(a)
        return result

