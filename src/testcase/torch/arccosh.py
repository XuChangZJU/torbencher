
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arccosh)
class TorchArccoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arccosh(self):
        a = torch.randn(4).uniform_(1, 10)
        result = torch.arccosh(a)
        return result


