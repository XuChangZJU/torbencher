
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReflectionPad2d)
class TorchNNReflectionPad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reflection_pad2d(self):
        a = torch.randn(1, 2, 4, 4)
        pad = torch.nn.ReflectionPad2d(2)
        result = pad(a)
        return result

