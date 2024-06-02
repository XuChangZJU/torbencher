
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_max_pool3d)
class TorchNNFunctionalAdaptiveMaxPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool3d_5d(self, input=None):
        if input is not None:
            result = torch.nn.functional.adaptive_max_pool3d(input[0], input[1])
            return [result, input]
        a = torch.randn(1, 8, 6, 4, 2)
        b = (2, 2, 2)
        result = torch.nn.functional.adaptive_max_pool3d(a, b)
        return [result, [a, b]]


