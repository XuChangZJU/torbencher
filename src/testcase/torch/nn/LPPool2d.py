
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LPPool2d)
class TorchNNLPPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool2d(self):
        a = torch.randn(1, 2, 4, 4)
        pool = torch.nn.LPPool2d(2, 3)
        result = pool(a)
        return result

