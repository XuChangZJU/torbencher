
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sort)
class TorchSortTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sort(self):
        a = torch.randn(4, 4)
        result = torch.sort(a, 1)
        return result

