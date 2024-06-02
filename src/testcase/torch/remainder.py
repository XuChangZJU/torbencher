
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.remainder)
class TorchRemainderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remainder_number(self, input=None):
        if input is not None:
            result = torch.remainder(input[0], input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4, 5])
        result = torch.remainder(a, 3)
        return [result, [a, 3]]

    @test_api_version.larger_than("1.1.3")
    def test_remainder(self, input=None):
        if input is not None:
            result = torch.remainder(input[0], input[1])
            return [result, input]
        a = torch.tensor([-3, -2, -1, 1, 2, 3])
        b = torch.tensor([2, 2, 2, 2, 2, 2])
        result = torch.remainder(a, b)
        return [result, [a, b]]

