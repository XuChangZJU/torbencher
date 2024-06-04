
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dist)
class TorchDistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dist(self):
        
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.dist(a, b, p=2)
        return result

