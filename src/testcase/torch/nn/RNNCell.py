
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RNNCell)
class TorchRNNCellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rnncell_correctness(self):
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        input_tensor = torch.randn(random.randint(1, 10), input_size)
        hidden = torch.randn(random.randint(1, 10), hidden_size)
        rnn_cell = torch.nn.RNNCell(input_size, hidden_size)
        result = rnn_cell(input_tensor, hidden)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rnncell_large_scale(self):
        input_size = random.randint(100, 1000)
        hidden_size = random.randint(100, 1000)
        input_tensor = torch.randn(random.randint(1000, 10000), input_size)
        hidden = torch.randn(random.randint(1000, 10000), hidden_size)
        rnn_cell = torch.nn.RNNCell(input_size, hidden_size)
        result = rnn_cell(input_tensor, hidden)
        return result

