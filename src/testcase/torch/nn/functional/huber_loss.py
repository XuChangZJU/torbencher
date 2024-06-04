
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.huber_loss)
class TorchNNFunctionalHuberLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_huber_loss(self):
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.nn.functional.huber_loss(a, b)
        return result


