
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.xlogy)
class TorchSpecialXlogyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_xlogy_0d(self):
        
        a = torch.randn([])
        b = torch.randn([])
        result = torch.special.xlogy(a, b)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_xlogy_1d_0d(self):
        
        a = torch.randn(5)
        b = torch.randn([])
        result = torch.special.xlogy(a, b)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_xlogy_0d_1d(self):
        
        a = torch.randn([])
        b = torch.randn(5)
        result = torch.special.xlogy(a, b)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_xlogy_1d(self):
        
        a = torch.randn(5)
        b = torch.randn(5)
        result = torch.special.xlogy(a, b)
        return result


