
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveAvgPool1d)
class TorchNNAdaptiveAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool1d(self, input=None):
        if input is not None:
            result = torch.nn.AdaptiveAvgPool1d(input[0])(input[1])
            return [result, input]
        a = torch.randn(1, 10)
        pool = torch.nn.AdaptiveAvgPool1d(5)
        result = pool(a)
        return [result, [5, a]]

