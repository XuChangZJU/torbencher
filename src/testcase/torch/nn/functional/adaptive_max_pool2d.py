
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_max_pool2d)
class TorchNNFunctionalAdaptiveMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool2d_4d(self):
        
        a = torch.randn(1, 8, 6, 4)
        b = (2, 2)
        result = torch.nn.functional.adaptive_max_pool2d(a, b)
        return result


