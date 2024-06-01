import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.where)
class TorchWhereTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_where_4d(self, input=None):
        if input is not None:
            result = torch.where(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        result = torch.where(a > 0, b, c)
        return [result, [a, b, c]]

