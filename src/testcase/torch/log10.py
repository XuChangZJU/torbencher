
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.log10)
class TorchLog10TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log10(self):
        a = torch.randn(5)
        result = torch.log10(a)
        return result

