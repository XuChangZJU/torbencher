
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.pinv)
class TorchLinalgPinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_pinv(self, input=None):
        if input is not None:
            result = torch.linalg.pinv(input[0])
            return [result, input]
        a = torch.randn(3, 3)
        result = torch.linalg.pinv(a)
        return [result, [a]]

