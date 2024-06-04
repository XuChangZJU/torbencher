
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isclose)
class TorchIscloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isclose(self):
        
        a = torch.tensor([10000., 1e-07])
        b = torch.tensor([10000.1, 1e-08])
        result = torch.isclose(a, b, rtol=1e-05, atol=1e-08)
        return result

