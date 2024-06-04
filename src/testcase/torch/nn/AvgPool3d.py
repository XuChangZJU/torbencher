
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AvgPool3d)
class TorchNNAvgPool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool3d(self):
        
        a = torch.randn(1, 10, 10, 10)
        pool = torch.nn.AvgPool3d(3)
        result = pool(a)
        return result

