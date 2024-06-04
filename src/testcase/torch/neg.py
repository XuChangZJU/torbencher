
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.neg)
class TorchNegTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_neg(self):
        
        a = torch.randn(5)
        result = torch.neg(a)
        return result

