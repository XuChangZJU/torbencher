
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diag)
class TorchDiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diag(self):
        
        a = torch.randn(3, 3)
        result = torch.diag(a, diagonal=1)
        return result

