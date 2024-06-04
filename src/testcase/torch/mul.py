
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mul)
class TorchMulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mul_number(self):
        
        a = torch.randn(4)
        result = torch.mul(a, 10)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_mul(self):
        
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.mul(a, b)
        return result

