
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LSTM)
class TorchNNLSTMTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lstm(self, input=None):
        if input is not None:
            result = torch.nn.LSTM(input[0], input[1], input[2])(input[3])
            return [result, input]
        a = torch.randn(5, 3, 10)
        lstm = torch.nn.LSTM(10, 20, 2)
        result = lstm(a)
        return [result, [10, 20, 2, a]]

