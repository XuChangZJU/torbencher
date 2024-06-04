
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmax)
class TorchNNSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax(self):
        
        a = torch.randn(10)
        softmax = torch.nn.Softmax(dim=1)
        result = softmax(a)
        return result

