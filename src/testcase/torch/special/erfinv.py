
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.erfinv)
class TorchSpecialErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_erfinv_0d(self):
        a = torch.rand([])
        result = torch.special.erfinv(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_erfinv_1d(self):
        a = torch.rand(5)
        result = torch.special.erfinv(a)
        return result

