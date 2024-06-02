
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.i1)
class TorchSpecialI1TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_i1_0d(self, input=None):
        if input is not None:
            result = torch.special.i1(input[0])
            return [result, input]
        a = torch.randn([])
        result = torch.special.i1(a)
        return [result, [a]]

    @test_api_version.larger_than("1.1.3")
    def test_i1_1d(self, input=None):
        if input is not None:
            result = torch.special.i1(input[0])
            return [result, input]
        a = torch.randn(5)
        result = torch.special.i1(a)
        return [result, [a]]

