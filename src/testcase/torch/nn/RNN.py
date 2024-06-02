
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNN)
class TorchNNRNNTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnn(self, input=None):
        if input is not None:
            result = torch.nn.RNN(input[0], input[1], input[2])(input[3])
            return [result, input]
        a = torch.randn(5, 3, 10)
        rnn = torch.nn.RNN(10, 20, 2)
        result = rnn(a)
        return [result, [10, 20, 2, a]]

