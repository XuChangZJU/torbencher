
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logical_or)
class TorchLogicalOrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_or(self):
        a = torch.tensor([True, False, True])
        b = torch.tensor([False, False, True])
        result = torch.logical_or(a, b)
        return result

