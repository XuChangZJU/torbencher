
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Tanhshrink)
class TorchNNTanhshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tanhshrink(self):
        a = torch.randn(10)
        tanhshrink = torch.nn.Tanhshrink()
        result = tanhshrink(a)
        return result

