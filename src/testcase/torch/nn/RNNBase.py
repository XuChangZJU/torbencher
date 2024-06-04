
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNBase)
class TorchNNRNNBaseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnn_base(self):
        a = torch.randn(5, 3, 10)
        rnn = torch.nn.RNNBase(2, 10, 20, 2, 0.5)
        result = rnn(a)
        return result

