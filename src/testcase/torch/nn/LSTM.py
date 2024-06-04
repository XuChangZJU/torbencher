
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LSTM)
class TorchNNLSTMTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lstm(self):
        
        a = torch.randn(5, 3, 10)
        lstm = torch.nn.LSTM(10, 20, 2)
        result = lstm(a)
        return result

