
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmax2d)
class TorchNNSoftmax2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax2d(self):
        a = torch.randn(1, 5, 5, 5)
        softmax = torch.nn.Softmax2d()
        result = softmax(a)
        return result

