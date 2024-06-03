
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveAvgPool2d)
class TorchNNAdaptiveAvgPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool2d(self, input=None):
        if input is not None:
            result = torch.nn.AdaptiveAvgPool2d(input[0])(input[1])
            return result
        a = torch.randn(1, 10, 10)
        pool = torch.nn.AdaptiveAvgPool2d((5, 5))
        result = pool(a)
        return result

