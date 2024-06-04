
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logaddexp2)
class TorchLogaddexp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logaddexp2(self):
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.logaddexp2(a, b)
        return result


