import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arange)
class TorchArangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arange_4d(self, input=None):
        if input is not None:
            result = torch.arange(input[0], input[1], input[2])
            return [result, input]
        result = torch.arange(0, 10, 2)
        return [result, [0, 10, 2]]

