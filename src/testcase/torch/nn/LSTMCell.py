import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.LSTMCell)
class TorchNnLstmcellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_LSTMCell_correctness(self):
        # Define random dimensions for input, hidden state, and batch size
        batch_size = random.randint(1, 10)
        input_size = random.randint(1, 20)
        hidden_size = random.randint(1, 20)

        # Create random input tensor
        input_tensor = torch.randn(batch_size, input_size)

        # Create random initial hidden state and cell state
        hx = torch.randn(batch_size, hidden_size)
        cx = torch.randn(batch_size, hidden_size)

        # Create LSTMCell instance
        lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

        # Pass input, hidden state, and cell state through LSTM cell
        hx_next, cx_next = lstm_cell(input_tensor, (hx, cx))

        # Return the next hidden state and cell state
        return hx_next, cx_next
