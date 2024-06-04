
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atan2)
class TorchAtan2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atan2(self):
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.atan2(a, b)
        return result


