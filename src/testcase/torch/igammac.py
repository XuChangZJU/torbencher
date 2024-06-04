
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.igammac)
class TorchIgammacTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_igammac_0d(self):
        
        a = torch.randn(()) + 1
        b = torch.randn(()) + 1
        result = torch.igammac(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_igammac_1d(self):
        
        a = torch.randn(4) + 1
        b = torch.randn(4) + 1
        result = torch.igammac(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_igammac_2d(self):
        
        a = torch.randn(4, 4) + 1
        b = torch.randn(4, 4) + 1
        result = torch.igammac(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_igammac_3d(self):
        
        a = torch.randn(4, 4, 4) + 1
        b = torch.randn(4, 4, 4) + 1
        result = torch.igammac(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_igammac_4d(self):
        
        a = torch.randn(4, 4, 4, 4) + 1
        b = torch.randn(4, 4, 4, 4) + 1
        result = torch.igammac(a, b)
        return result


