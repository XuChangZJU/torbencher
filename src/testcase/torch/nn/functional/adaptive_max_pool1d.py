
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_max_pool1d)
class TorchNNFunctionalAdaptiveMaxPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool1d_3d(self):
        
        a = torch.randn(1, 8, 6)
        b = 4
        result = torch.nn.functional.adaptive_max_pool1d(a, b)
        return result


