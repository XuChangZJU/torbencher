
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.add)
class TorchAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add(self, input=None):
        if input is not None:
            result = torch.add(input[0], input[1], input[2])
            return result
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.add(a, b, alpha=10)
        return result

