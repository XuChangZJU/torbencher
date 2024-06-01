import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_avg_pool3d)
class TorchNNFunctionalAdaptiveAvgPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool3d_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.adaptive_avg_pool3d(input[0], input[1])
            return [result, input]
        a = torch.randn(2, 3, 8, 8, 8)
        b = (4, 4, 4)
        result = torch.nn.functional.adaptive_avg_pool3d(a, b)
        return [result, [a, b]]

