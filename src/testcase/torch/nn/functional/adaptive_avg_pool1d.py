
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_avg_pool1d)
class TorchNNFunctionalAdaptiveAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool1d_3d(self, input=None):
        if input is not None:
            result = torch.nn.functional.adaptive_avg_pool1d(input[0], input[1])
            return [result, input]
        a = torch.randn(1, 8, 6)
        b = 4
        result = torch.nn.functional.adaptive_avg_pool1d(a, b)
        return [result, [a, b]]


