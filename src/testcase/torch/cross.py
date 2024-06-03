
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cross)
class TorchCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross(self, input=None):
        if input is not None:
            result = torch.cross(input[0], input[1])
            return result
        a = torch.randn(4, 3)
        b = torch.randn(4, 3)
        result = torch.cross(a, b, dim=1)
        return result

