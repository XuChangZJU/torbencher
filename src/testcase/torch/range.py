import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.range)
class TorchRangeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_range_4d(self, input=None):
        if input is not None:
            result = torch.range(input[0], input[1], input[2])
            return [result, input]
        result = torch.range(0, 10, 2)
        return [result, [0, 10, 2]]

