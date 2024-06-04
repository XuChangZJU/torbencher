
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.deg2rad)
class TorchDeg2radTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_deg2rad(self):
        
        a = torch.randn(4)
        result = torch.deg2rad(a)
        return result


