
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AvgPool1d)
class TorchNNAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d(self):
        a = torch.randn(1, 10)
        pool = torch.nn.AvgPool1d(3)
        result = pool(a)
        return result

