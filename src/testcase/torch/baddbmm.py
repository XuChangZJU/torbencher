
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.baddbmm)
class TorchBaddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_baddbmm(self):
        a = torch.randn(10, 3, 5)
        b = torch.randn(10, 3, 4)
        c = torch.randn(10, 4, 5)
        result = torch.baddbmm(a, b, c, beta = 0.2, alpha=5)
        return result

