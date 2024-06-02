
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lt)
class TorchLtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lt_number(self, input=None):
        if input is not None:
            result = torch.lt(input[0], input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3])
        result = torch.lt(a, 2)
        return [result, [a, 2]]

    @test_api_version.larger_than("1.1.3")
    def test_lt(self, input=None):
        if input is not None:
            result = torch.lt(input[0], input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([0, 2, 4])
        result = torch.lt(a, b)
        return [result, [a, b]]

