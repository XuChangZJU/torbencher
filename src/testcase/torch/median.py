
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.median)
class TorchMedianTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_median_dim(self):
        
        a = torch.randn(4, 4)
        result = torch.median(a, 1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_median(self):
        
        a = torch.randn(4, 4)
        result = torch.median(a)
        return result

