import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gt)
class TorchGtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gt_4d(self, input=None):
        if input is not None:
            result = torch.gt(input[0], input[1])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.gt(a, b)
        return [result, [a, b]]

