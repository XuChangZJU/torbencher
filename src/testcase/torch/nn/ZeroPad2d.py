
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ZeroPad2d)
class TorchNNZeroPad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zero_pad2d(self, input=None):
        if input is not None:
            result = torch.nn.ZeroPad2d(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 2, 4, 4)
        pad = torch.nn.ZeroPad2d(2)
        result = pad(a)
        return [result, [2, a]]
