
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.absolute)
class TorchAbsoluteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_absolute(self, input=None):
        if input is not None:
            result = torch.absolute(input[0])
            return result
        a = torch.randn(4)
        result = torch.absolute(a)
        return result


