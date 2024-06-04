
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cumsum)
class TorchCumsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumsum(self):
        
        a = torch.randn(10)
        result = torch.cumsum(a, dim=0)
        return result

