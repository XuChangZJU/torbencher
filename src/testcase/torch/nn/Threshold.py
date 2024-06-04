
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Threshold)
class TorchNNThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold(self):
        a = torch.randn(10)
        threshold = torch.nn.Threshold(0.1, 20)
        result = threshold(a)
        return result

