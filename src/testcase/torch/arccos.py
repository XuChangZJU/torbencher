
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arccos)
class TorchArccosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arccos(self, input=None):
        if input is not None:
            result = torch.arccos(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.arccos(a)
        return [result, [a]]


