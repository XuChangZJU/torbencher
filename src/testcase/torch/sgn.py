
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sgn)
class TorchSgnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sgn(self):
        
        a = torch.randn(4)
        result = torch.sgn(a)
        return result


