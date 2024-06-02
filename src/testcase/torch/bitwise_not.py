
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_not)
class TorchBitwiseNotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_not(self, input=None):
        if input is not None:
            result = torch.bitwise_not(input[0])
            return [result, input]
        a = torch.tensor([True, False])
        result = torch.bitwise_not(a)
        return [result, [a]]


