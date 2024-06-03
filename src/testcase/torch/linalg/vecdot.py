
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vecdot)
class TorchLinalgVecdotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11.0")
    def test_vecdot(self, input=None):
        if input is not None:
            result = torch.linalg.vecdot(input[0], input[1])
            return result
        a = torch.randn(3)
        b = torch.randn(3)
        result = torch.linalg.vecdot(a, b)
        return result
