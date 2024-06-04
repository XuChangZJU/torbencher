
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.exp2)
class TorchSpecialExp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exp2_0d(self):
        a = torch.randn([])
        result = torch.special.exp2(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_exp2_1d(self):
        a = torch.randn(5)
        result = torch.special.exp2(a)
        return result

