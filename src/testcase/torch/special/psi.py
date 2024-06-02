
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.psi)
class TorchSpecialPsiTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_psi_0d(self, input=None):
        if input is not None:
            result = torch.special.psi(input[0])
            return [result, input]
        a = torch.randn([])
        result = torch.special.psi(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_psi_1d(self, input=None):
        if input is not None:
            result = torch.special.psi(input[0])
            return [result, input]
        a = torch.randn(5)
        result = torch.special.psi(a)
        return [result, [a]]

