import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.LSTMCell)
class TorchNnLstmcellTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    # @unittest.skip
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

        with torch.no_grad():
            # Use random.randint to set a random range for weight initialization
            lstm_cell.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
            lstm_cell.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)

            # Initialize bias with random.normal
            lstm_cell.bias_ih = torch.nn.Parameter(torch.normal(0.0, 0.01, (4 * hidden_size,)))
            lstm_cell.bias_hh = torch.nn.Parameter(torch.normal(0.0, 0.01, (4 * hidden_size,)))

        # Pass input, hidden state, and cell state through LSTM cell
        hx_next, cx_next = lstm_cell(input_tensor, (hx, cx))

        # Return the next hidden state and cell state
        return hx_next, cx_next
