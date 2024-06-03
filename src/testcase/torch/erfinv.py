
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.erfinv)
class TorchErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_erfinv_0d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return result
        a = torch.rand(())
        result = torch.erfinv(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_1d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return result
        a = torch.rand(4)
        result = torch.erfinv(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_2d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return result
        a = torch.rand(4, 4)
        result = torch.erfinv(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_3d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return result
        a = torch.rand(4, 4, 4)
        result = torch.erfinv(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_4d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return result
        a = torch.rand(4, 4, 4, 4)
        result = torch.erfinv(a)
        return result
