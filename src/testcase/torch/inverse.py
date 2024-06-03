
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.inverse)
class TorchInverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inverse(self, input=None):
        if input is not None:
            result = torch.inverse(input[0])
            return result
        a = torch.randn(4, 4)
        result = torch.inverse(a)
        return result

