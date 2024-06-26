
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_fill)
class TorchIndexFillTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_fill_1d(self, input=None):
        if input is not None:
            result = torch.index_fill(input[0], dim=input[1], index=input[2], value=input[3])
            return [result, input]
        a = torch.randn(5, 3)
        dim = 0
        index = torch.tensor([0, 2])
        value = 10
        result = torch.index_fill(a, dim=dim, index=index, value=value)
        return [result, [a, dim, index, value]]

# torch.index_select
