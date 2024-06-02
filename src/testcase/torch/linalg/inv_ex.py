
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.inv_ex)
class TorchLinalgInvExTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_inv_ex(self, input=None):
        if input is not None:
            result = torch.linalg.inv_ex(input[0])
            return [result, input]
        a = torch.randn(3, 3)
        result = torch.linalg.inv_ex(a)
        return [result, [a]]

