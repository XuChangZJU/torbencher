
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.trunc)
class TorchTruncTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trunc(self):
        
        a = torch.randn(4)
        result = torch.trunc(a)
        return result

