
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atanh)
class TorchAtanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atanh_0d(self):
        a = torch.randn(())
        result = torch.atanh(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atanh_1d(self):
        a = torch.randn(4)
        result = torch.atanh(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atanh_2d(self):
        a = torch.randn(4, 4)
        result = torch.atanh(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atanh_3d(self):
        a = torch.randn(4, 4, 4)
        result = torch.atanh(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atanh_4d(self):
        a = torch.randn(4, 4, 4, 4)
        result = torch.atanh(a)
        return result

