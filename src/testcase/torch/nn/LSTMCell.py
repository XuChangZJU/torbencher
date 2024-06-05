
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LSTMCell)
class TorchLSTMCellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lstmcell_correctness(self):
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), input_size)
        hx = (torch.randn(random.randint(1, 10), hidden_size), torch.randn(random.randint(1, 10), hidden_size))
        lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        result = lstm_cell(input_tensor, hx)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_lstmcell_large_scale(self):
        input_size = random.randint(100, 1000)
        hidden_size = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), input_size)
        hx = (torch.randn(random.randint(1000, 10000), hidden_size), torch.randn(random.randint(1000, 10000), hidden_size))
        lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        result = lstm_cell(input_tensor, hx)
        return result

