
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNBase)
class TorchNNRNNBaseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnn_base(self, input=None):
        if input is not None:
            result = torch.nn.RNNBase(input[0], input[1], input[2], input[3], input[4])(input[5])
            return [result, input]
        a = torch.randn(5, 3, 10)
        rnn = torch.nn.RNNBase(2, 10, 20, 2, 0.5)
        result = rnn(a)
        return [result, [2, 10, 20, 2, 0.5, a]]

