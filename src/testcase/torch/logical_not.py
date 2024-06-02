
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logical_not)
class TorchLogicalNotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_not(self, input=None):
        if input is not None:
            result = torch.logical_not(input[0])
            return [result, input]
        a = torch.tensor([True, False, True])
        result = torch.logical_not(a)
        return [result, [a]]

