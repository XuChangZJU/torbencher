
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dot)
class TorchDotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dot(self):
        
        a = torch.randn(5)
        b = torch.randn(5)
        result = torch.dot(a, b)
        return result

