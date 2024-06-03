
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.i1e)
class TorchSpecialI1eTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_i1e_0d(self, input=None):
        if input is not None:
            result = torch.special.i1e(input[0])
            return result
        a = torch.randn([])
        result = torch.special.i1e(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_i1e_1d(self, input=None):
        if input is not None:
            result = torch.special.i1e(input[0])
            return result
        a = torch.randn(5)
        result = torch.special.i1e(a)
        return result

