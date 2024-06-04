
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Hardshrink)
class TorchNNHardshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardshrink(self):
        
        a = torch.randn(10)
        hardshrink = torch.nn.Hardshrink()
        result = hardshrink(a)
        return result

