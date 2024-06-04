
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_select)
class TorchIndexSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_select(self):
        a = torch.randn(4, 4)
        b = 1
        c = torch.tensor([0, 2])
        result = torch.index_select(a, b, c)
        return result

