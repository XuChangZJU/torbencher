
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool1d)
class TorchNNLPPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool1d(self):
        a = torch.randn(1, 2, 4)
        pool = torch.nn.LPPool1d(2, 3)
        result = pool(a)
        return result

