
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LogSoftmax)
class TorchNNLogSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_softmax(self):
        
        a = torch.randn(10)
        log_softmax = torch.nn.LogSoftmax(dim=1)
        result = log_softmax(a)
        return result

