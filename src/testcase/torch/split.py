import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.split)
class TorchSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_split_4d(self, input=None):
        if input is not None:
            result = torch.split(input[0], input[1])
            return [result, input]
        a = torch.randn(8)
        result = torch.split(a, 2)
        return [result, [a, 2]]

