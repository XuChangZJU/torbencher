
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lstsq)
class TorchLstsqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lstsq_4d(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 2)
        result = torch.lstsq(b, a)
        return result

