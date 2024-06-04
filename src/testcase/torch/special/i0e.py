
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.i0e)
class TorchSpecialI0eTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_i0e_0d(self):
        a = torch.randn([])
        result = torch.special.i0e(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_i0e_1d(self):
        a = torch.randn(5)
        result = torch.special.i0e(a)
        return result

