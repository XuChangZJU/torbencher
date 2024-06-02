
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.conj)
class TorchConjTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conj(self, input=None):
        if input is not None:
            result = torch.conj(input[0])
            return [result, input]
        a = torch.randn(4) + 1j * torch.randn(4)
        result = torch.conj(a)
        return [result, [a]]


