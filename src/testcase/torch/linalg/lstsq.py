
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.lstsq)
class TorchLinalgLstsqTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_lstsq(self):
        a = torch.randn(3, 2)
        b = torch.randn(3, 1)
        result = torch.linalg.lstsq(a, b)
        return result


