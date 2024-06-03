
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bartlett_window)
class TorchBartlettWindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bartlett_window(self, input=None):
        if input is not None:
            result = torch.bartlett_window(input[0])
            return result
        a = 10
        result = torch.bartlett_window(a)
        return result

