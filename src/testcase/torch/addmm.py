
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmm)
class TorchAddmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmm(self):
        
        a = torch.randn(2, 3)
        b = torch.randn(5, 2, 4)
        c = torch.randn(5, 4, 3)
        result = torch.addmm(a, b, c, beta = 0.2, alpha=5)
        return result

