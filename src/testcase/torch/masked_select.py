import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.masked_select)
class TorchMaskedSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_masked_select_4d(self, input=None):
        if input is not None:
            result = torch.masked_select(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        b = a > 0
        result = torch.masked_select(a, b)
        return [result, [a, b]]

