
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softshrink)
class TorchNNSoftshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softshrink(self, input=None):
        if input is not None:
            result = torch.nn.Softshrink()(input[0])
            return [result, input]
        a = torch.randn(10)
        softshrink = torch.nn.Softshrink()
        result = softshrink(a)
        return [result, [a]]

