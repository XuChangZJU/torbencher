
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arange)
class TorchArangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arange_one_param(self, input=None):
        if input is not None:
            result = torch.arange(input[0])
            return [result, input]
        result = torch.arange(5)
        return [result, [5]]
    @test_api_version.larger_than("1.1.3")
    def test_arange(self, input=None):
        if input is not None:
            result = torch.arange(input[0], input[1], input[2])
            return [result, input]
        result = torch.arange(1, 2.5, 0.5)
        return [result, [1, 2.5, 0.5]]

