
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.clamp)
class TorchClampTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp_min(self):
        
        a = torch.randn(4)
        result = torch.clamp(a, min=0)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clamp_max(self):
        
        a = torch.randn(4)
        result = torch.clamp(a, max=0)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_clamp(self):
        
        a = torch.randn(4)
        result = torch.clamp(a, min=-0.5, max=0.5)
        return result

