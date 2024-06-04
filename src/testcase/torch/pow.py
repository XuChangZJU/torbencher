
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.pow)
class TorchPowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pow(self):
        
        a = torch.randn(4).uniform_(0, 10)
        b = torch.randn(4).uniform_(0, 2)
        result = torch.pow(a, b)
        return result


