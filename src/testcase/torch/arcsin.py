
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arcsin)
class TorchArcsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arcsin(self, input=None):
        if input is not None:
            result = torch.arcsin(input[0])
            return result
        a = torch.randn(4).uniform_(-1, 1)
        result = torch.arcsin(a)
        return result


