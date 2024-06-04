
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.log1p)
class TorchLog1pTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log1p(self):
        a = torch.randn(4)
        result = torch.log1p(a)
        return result


