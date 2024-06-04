
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.add)
class TorchAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_10d(self):
        a = torch.randn(10)
        b = torch.randn(10)
        result = torch.add(a, b, alpha=10)
        return result
    
    def test_add_100d(self):
        a = torch.randn(100)
        b = torch.randn(100)
        result = torch.add(a, b, alpha=10)
        return result

