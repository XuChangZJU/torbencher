
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.gt)
class TorchGtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gt_number(self, input=None):
        if input is not None:
            result = torch.gt(input[0], input[1])
            return result
        a = torch.tensor([1, 2, 3])
        result = torch.gt(a, 2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_gt(self, input=None):
        if input is not None:
            result = torch.gt(input[0], input[1])
            return result
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([0, 2, 4])
        result = torch.gt(a, b)
        return result

