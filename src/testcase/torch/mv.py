
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mv)
class TorchMvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mv_4d(self):
        a = torch.randn(4, 5)
        b = torch.randn(5)
        result = torch.mv(a, b)
        return result

