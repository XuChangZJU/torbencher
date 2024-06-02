
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.erfinv)
class TorchSpecialErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_erfinv_0d(self, input=None):
        if input is not None:
            result = torch.special.erfinv(input[0])
            return [result, input]
        a = torch.rand([])
        result = torch.special.erfinv(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_1d(self, input=None):
        if input is not None:
            result = torch.special.erfinv(input[0])
            return [result, input]
        a = torch.rand(5)
        result = torch.special.erfinv(a)
        return [result, [a]]

