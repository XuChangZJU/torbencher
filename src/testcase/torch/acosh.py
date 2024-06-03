
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.acosh)
class TorchAcoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_acosh(self, input=None):
        if input is not None:
            result = torch.acosh(input[0])
            return result
        a = torch.randn(4).uniform_(1, 10)
        result = torch.acosh(a)
        return result


