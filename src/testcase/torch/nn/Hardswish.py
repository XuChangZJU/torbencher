
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Hardswish)
class TorchNNHardswishTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardswish(self):
        a = torch.randn(10)
        hardswish = torch.nn.Hardswish()
        result = hardswish(a)
        return result

